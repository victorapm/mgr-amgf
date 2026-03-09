#include <stdio.h>
#include <stdlib.h>

#include "amgf_mgr_minimal.h"

/*
 * Minimal AMGF via hypre MGR:
 * - 2-level MGR (one elimination level + coarse level)
 * - AMG as level-0 G-relaxation
 * - SuperLU_Dist as coarsest solver
 *
 * Constraint mask convention (local rows):
 * - 1 -> C-point
 * - 0 -> F-point
 */

typedef struct AMGF_MGRData_struct
{
   MPI_Comm     comm;
   HYPRE_Int    local_num_rows;
   HYPRE_Int    mask_is_set;
   HYPRE_Int    print_level;
   HYPRE_Int   *point_marker_array;
   HYPRE_Solver mgr;
   HYPRE_Solver g_relax_amg;
   HYPRE_Solver coarse_direct;
} AMGF_MGRData;

static HYPRE_Int
AMGF_MGRPrintError(MPI_Comm comm, const char *msg)
{
   int myid = 0;
   MPI_Comm_rank(comm, &myid);
   if (myid == 0)
   {
      fprintf(stderr, "%s\n", msg);
   }
   return 1;
}

static HYPRE_Int
AMGF_MGRCheckSuperLUDist(MPI_Comm comm)
{
#ifndef HYPRE_USING_DSUPERLU
   return AMGF_MGRPrintError(
      comm,
      "AMGF_MGR requires hypre built with SuperLU_Dist (HYPRE_USING_DSUPERLU)."
   );
#else
   (void)comm;
   return 0;
#endif
}

#define HYPRE_RETURN_IF_ERROR(call) \
   do                               \
   {                                \
      HYPRE_Int ierr_ = (call);     \
      if (ierr_)                    \
      {                             \
         return ierr_;              \
      }                             \
   } while (0)

HYPRE_Int
AMGF_MGRCreate(MPI_Comm comm, HYPRE_Solver *solver)
{
   AMGF_MGRData *data = NULL;

   if (!solver)
   {
      return 1;
   }

   data = (AMGF_MGRData *)calloc(1, sizeof(*data));
   if (!data)
   {
      return 1;
   }

   data->comm = comm;
   data->print_level = 0;
   if (HYPRE_MGRCreate(&data->mgr))
   {
      free(data);
      return 1;
   }

   *solver = (HYPRE_Solver)data;
   return 0;
}

HYPRE_Int
AMGF_MGRSetConstraintMask(HYPRE_Solver solver, const HYPRE_Int *constraint_mask,
                          HYPRE_Int local_num_rows)
{
   AMGF_MGRData *data = (AMGF_MGRData *)solver;
   HYPRE_Int    *new_marker_array = NULL;

   if (!data || !constraint_mask || local_num_rows < 1)
   {
      return 1;
   }

   new_marker_array =
      (HYPRE_Int *)malloc((size_t)local_num_rows * sizeof(HYPRE_Int));
   if (!new_marker_array)
   {
      return 1;
   }

   for (HYPRE_Int i = 0; i < local_num_rows; i++)
   {
      new_marker_array[i] = (constraint_mask[i] != 0) ? 1 : 0;
   }

   free(data->point_marker_array);
   data->point_marker_array = new_marker_array;
   data->local_num_rows = local_num_rows;
   data->mask_is_set    = 1;
   return 0;
}

HYPRE_Int
AMGF_MGRSetPrintLevel(HYPRE_Solver solver, HYPRE_Int print_level)
{
   AMGF_MGRData *data = (AMGF_MGRData *)solver;

   if (!data || print_level < 0)
   {
      return 1;
   }

   data->print_level = print_level;
   return 0;
}

HYPRE_Int
AMGF_MGRSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
              HYPRE_ParVector x)
{
   AMGF_MGRData *data = (AMGF_MGRData *)solver;

   HYPRE_Int  num_c_dofs[1]         = {1};
   HYPRE_Int  c_dofs_level0[1]      = {1};
   HYPRE_Int *c_dofs_ptrs[1]        = {c_dofs_level0};
   HYPRE_Int  interp_type[1]        = {0};
   HYPRE_Int  restrict_type[1]      = {0};
   HYPRE_Int  coarse_grid_method[1] = {0};

   HYPRE_BigInt ilower = 0;
   HYPRE_BigInt iupper = -1;
   HYPRE_BigInt jl     = 0;
   HYPRE_BigInt ju     = -1;

   HYPRE_Int local_rows  = 0;
   HYPRE_Int local_cpts  = 0;
   HYPRE_Int global_rows = 0;
   HYPRE_Int global_cpts = 0;

   if (!data || !A || !b || !x)
   {
      return 1;
   }

   if (!data->mask_is_set)
   {
      return AMGF_MGRPrintError(
         data->comm, "AMGF_MGRSetup requires AMGF_MGRSetConstraintMask first."
      );
   }

   if (AMGF_MGRCheckSuperLUDist(data->comm))
   {
      return 1;
   }

   HYPRE_RETURN_IF_ERROR(HYPRE_ParCSRMatrixGetLocalRange(A, &ilower, &iupper, &jl, &ju));
   (void)jl;
   (void)ju;
   local_rows = (iupper >= ilower) ? (HYPRE_Int)(iupper - ilower + 1) : 0;

   if (local_rows != data->local_num_rows)
   {
      return AMGF_MGRPrintError(
         data->comm,
         "AMGF_MGR: local constraint-mask size does not match local matrix rows."
      );
   }

   for (HYPRE_Int i = 0; i < local_rows; i++)
   {
      local_cpts += data->point_marker_array[i];
   }

   if (MPI_Allreduce(&local_rows, &global_rows, 1, MPI_INT, MPI_SUM, data->comm) != MPI_SUCCESS ||
       MPI_Allreduce(&local_cpts, &global_cpts, 1, MPI_INT, MPI_SUM, data->comm) != MPI_SUCCESS)
   {
      return 1;
   }

   if (global_cpts <= 0 || global_cpts >= global_rows)
   {
      return AMGF_MGRPrintError(
         data->comm,
         "AMGF_MGR: invalid global C/F split (need at least one C and one F row)."
      );
   }

   HYPRE_RETURN_IF_ERROR(HYPRE_MGRSetCpointsByPointMarkerArray(data->mgr, 2, 1,
                                                                num_c_dofs,
                                                                c_dofs_ptrs,
                                                                data->point_marker_array));
   HYPRE_RETURN_IF_ERROR(HYPRE_MGRSetNonCpointsToFpoints(data->mgr, 1));
   HYPRE_RETURN_IF_ERROR(HYPRE_MGRSetMaxIter(data->mgr, 1));
   HYPRE_RETURN_IF_ERROR(HYPRE_MGRSetTol(data->mgr, 0.0));
   HYPRE_RETURN_IF_ERROR(HYPRE_MGRSetPrintLevel(data->mgr, data->print_level));
   HYPRE_RETURN_IF_ERROR(HYPRE_MGRSetMaxGlobalSmoothIters(data->mgr, 1));
   HYPRE_RETURN_IF_ERROR(HYPRE_MGRSetLevelInterpType(data->mgr, interp_type));
   HYPRE_RETURN_IF_ERROR(HYPRE_MGRSetLevelRestrictType(data->mgr, restrict_type));
   HYPRE_RETURN_IF_ERROR(HYPRE_MGRSetCoarseGridMethod(data->mgr, coarse_grid_method));

   if (!data->g_relax_amg)
   {
      HYPRE_RETURN_IF_ERROR(HYPRE_BoomerAMGCreate(&data->g_relax_amg));
      HYPRE_RETURN_IF_ERROR(HYPRE_BoomerAMGSetNumFunctions(data->g_relax_amg, 3));
      HYPRE_RETURN_IF_ERROR(HYPRE_BoomerAMGSetStrongThreshold(data->g_relax_amg, 0.5));
      HYPRE_RETURN_IF_ERROR(HYPRE_BoomerAMGSetRelaxType(data->g_relax_amg, 8));
      HYPRE_RETURN_IF_ERROR(HYPRE_BoomerAMGSetMaxIter(data->g_relax_amg, 1));
      HYPRE_RETURN_IF_ERROR(HYPRE_BoomerAMGSetTol(data->g_relax_amg, 0.0));
      HYPRE_RETURN_IF_ERROR(HYPRE_BoomerAMGSetPrintLevel(data->g_relax_amg, data->print_level));
   }

   HYPRE_RETURN_IF_ERROR(HYPRE_MGRSetGlobalSmootherAtLevel(data->mgr,
                                                            data->g_relax_amg, 0));

#ifdef HYPRE_USING_DSUPERLU
   if (!data->coarse_direct)
   {
      HYPRE_RETURN_IF_ERROR(HYPRE_MGRDirectSolverCreate(&data->coarse_direct));
      if (!data->coarse_direct)
      {
         return 1;
      }
   }

   HYPRE_RETURN_IF_ERROR(HYPRE_MGRSetCoarseSolver(data->mgr,
                                                   HYPRE_MGRDirectSolverSolve,
                                                   HYPRE_MGRDirectSolverSetup,
                                                   data->coarse_direct));
#else
   return 1;
#endif

   return HYPRE_MGRSetup(data->mgr, A, b, x);
}

HYPRE_Int
AMGF_MGRSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
              HYPRE_ParVector x)
{
   AMGF_MGRData *data = (AMGF_MGRData *)solver;

   if (!data || !A || !b || !x)
   {
      return 1;
   }

   return HYPRE_MGRSolve(data->mgr, A, b, x);
}

HYPRE_Int
AMGF_MGRDestroy(HYPRE_Solver solver)
{
   AMGF_MGRData *data = (AMGF_MGRData *)solver;

   if (!data)
   {
      return 0;
   }

   if (data->mgr)
   {
      HYPRE_MGRDestroy(data->mgr);
      data->mgr = NULL;
   }

#ifdef HYPRE_USING_DSUPERLU
   if (data->coarse_direct)
   {
      HYPRE_MGRDirectSolverDestroy(data->coarse_direct);
      data->coarse_direct = NULL;
   }
#endif

   free(data->point_marker_array);
   free(data);
   return 0;
}

#undef HYPRE_RETURN_IF_ERROR
