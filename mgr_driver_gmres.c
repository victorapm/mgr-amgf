#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HYPRE.h"
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "amgf_mgr_minimal.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct DriverOptions_struct
{
   char   matrix_prefix[PATH_MAX];
   char   rhs_prefix[PATH_MAX];
   char   dofmap_prefix[PATH_MAX];
   char   x0_prefix[PATH_MAX];
   int    has_x0;
   int    max_iter;
   int    k_dim;
   int    print_level;
   int    mgr_print_level;
   int    logging;
   double tol;
} DriverOptions;

static void
PrintMatrixHeader(int myid, int num_procs, const char *matrix_prefix,
                  HYPRE_BigInt global_rows, HYPRE_BigInt global_cols,
                  HYPRE_BigInt global_nnz, HYPRE_BigInt ilower,
                  HYPRE_BigInt iupper, HYPRE_BigInt jlower, HYPRE_BigInt jupper,
                  long long local_constraints, long long global_constraints)
{
   HYPRE_BigInt local_rows = 0;
   HYPRE_BigInt local_cols = 0;

   if (myid != 0)
   {
      return;
   }

   local_rows = (iupper >= ilower) ? (iupper - ilower + 1) : 0;
   local_cols = (jupper >= jlower) ? (jupper - jlower + 1) : 0;

   printf("\n");
   printf("============================================================\n");
   printf("AMGF-MGR Driver: Matrix Summary\n");
   printf("  MPI tasks: %d\n", num_procs);
   printf("  Matrix source: %s\n", matrix_prefix);
   printf("  Global size: %lld x %lld\n", (long long)global_rows,
          (long long)global_cols);
   printf("  Global nonzeros: %lld\n", (long long)global_nnz);
   printf("  Rank 0 local row range: [%lld, %lld] (%lld rows)\n",
          (long long)ilower, (long long)iupper, (long long)local_rows);
   printf("  Rank 0 local col range: [%lld, %lld] (%lld cols)\n",
          (long long)jlower, (long long)jupper, (long long)local_cols);
   printf("  Constraints (dofmap==1): rank0 local=%lld, global=%lld\n",
          local_constraints, global_constraints);
   printf("============================================================\n");
}

static void
DriverUsage(const char *prog)
{
   printf("Usage: %s [options]\n", prog);
   printf("\n");
   printf("Solve Ax=b with GMRES preconditioned by minimal AMGF-via-MGR.\n");
   printf("Matrix/vector are read with hypre IJ file readers.\n");
   printf("\n");
   printf("Options:\n");
   printf("  --matrix-prefix <path>   IJ matrix prefix (default: "
          "test/blocks-unsym-ls_00001/IJ.out.A)\n");
   printf("  --rhs-prefix <path>      IJ rhs prefix (default: "
          "test/blocks-unsym-ls_00001/IJ.out.b)\n");
   printf("  --dofmap-prefix <path>   dofmap prefix (default: "
          "test/blocks-unsym-ls_00001/dofmap.out)\n");
   printf("  --x0-prefix <path>       optional IJ initial guess prefix\n");
   printf("  --tol <real>             GMRES relative tolerance (default: 1e-8)\n");
   printf("  --max-iter <int>         GMRES max iterations (default: 200)\n");
   printf("  --k-dim <int>            GMRES Krylov dimension (default: 50)\n");
   printf("  --print-level <int>      hypre GMRES print level (default: 2)\n");
   printf("  --mgr-print-level <int>  hypre MGR print level (default: 0)\n");
   printf("  --logging <0|1>          hypre GMRES logging (default: 1)\n");
   printf("  -h, --help               print this help message\n");
}

static int
ParseIntArg(const char *name, const char *text, int *value_out)
{
   char *endptr = NULL;
   long  v      = 0;

   errno = 0;
   v     = strtol(text, &endptr, 10);
   if (errno || endptr == text || *endptr != '\0' || v < INT_MIN || v > INT_MAX)
   {
      fprintf(stderr, "Invalid integer for %s: %s\n", name, text);
      return 0;
   }

   *value_out = (int)v;
   return 1;
}

static int
ParseDoubleArg(const char *name, const char *text, double *value_out)
{
   char  *endptr = NULL;
   double v      = 0.0;

   errno = 0;
   v     = strtod(text, &endptr);
   if (errno || endptr == text || *endptr != '\0')
   {
      fprintf(stderr, "Invalid real value for %s: %s\n", name, text);
      return 0;
   }

   *value_out = v;
   return 1;
}

static int
DriverParseArgs(int argc, char **argv, DriverOptions *opts)
{
   for (int i = 1; i < argc; i++)
   {
      if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
      {
         DriverUsage(argv[0]);
         return 0;
      }
      else if (!strcmp(argv[i], "--matrix-prefix") && (i + 1 < argc))
      {
         snprintf(opts->matrix_prefix, sizeof(opts->matrix_prefix), "%s", argv[++i]);
      }
      else if (!strcmp(argv[i], "--rhs-prefix") && (i + 1 < argc))
      {
         snprintf(opts->rhs_prefix, sizeof(opts->rhs_prefix), "%s", argv[++i]);
      }
      else if (!strcmp(argv[i], "--dofmap-prefix") && (i + 1 < argc))
      {
         snprintf(opts->dofmap_prefix, sizeof(opts->dofmap_prefix), "%s", argv[++i]);
      }
      else if (!strcmp(argv[i], "--x0-prefix") && (i + 1 < argc))
      {
         snprintf(opts->x0_prefix, sizeof(opts->x0_prefix), "%s", argv[++i]);
         opts->has_x0 = 1;
      }
      else if (!strcmp(argv[i], "--tol") && (i + 1 < argc))
      {
         if (!ParseDoubleArg("--tol", argv[++i], &opts->tol))
         {
            return -1;
         }
      }
      else if (!strcmp(argv[i], "--max-iter") && (i + 1 < argc))
      {
         if (!ParseIntArg("--max-iter", argv[++i], &opts->max_iter))
         {
            return -1;
         }
      }
      else if (!strcmp(argv[i], "--k-dim") && (i + 1 < argc))
      {
         if (!ParseIntArg("--k-dim", argv[++i], &opts->k_dim))
         {
            return -1;
         }
      }
      else if (!strcmp(argv[i], "--print-level") && (i + 1 < argc))
      {
         if (!ParseIntArg("--print-level", argv[++i], &opts->print_level))
         {
            return -1;
         }
      }
      else if (!strcmp(argv[i], "--mgr-print-level") && (i + 1 < argc))
      {
         if (!ParseIntArg("--mgr-print-level", argv[++i], &opts->mgr_print_level))
         {
            return -1;
         }
      }
      else if (!strcmp(argv[i], "--logging") && (i + 1 < argc))
      {
         if (!ParseIntArg("--logging", argv[++i], &opts->logging))
         {
            return -1;
         }
      }
      else
      {
         fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
         DriverUsage(argv[0]);
         return -1;
      }
   }

   if (opts->max_iter < 1 || opts->k_dim < 1)
   {
      fprintf(stderr, "--max-iter and --k-dim must be positive integers\n");
      return -1;
   }
   if (opts->tol <= 0.0)
   {
      fprintf(stderr, "--tol must be > 0\n");
      return -1;
   }

   return 1;
}

static int
ReadLocalDofmap(MPI_Comm comm, const char *prefix, HYPRE_Int **dofmap_out,
                HYPRE_Int *local_size_out)
{
   int  myid = 0;
   FILE *fp  = NULL;
   char part_filename[PATH_MAX];
   size_t n = 0;
   HYPRE_Int *vals = NULL;

   *dofmap_out     = NULL;
   *local_size_out = 0;

   MPI_Comm_rank(comm, &myid);
   snprintf(part_filename, sizeof(part_filename), "%s.%05d", prefix, myid);

   fp = fopen(part_filename, "r");
   if (!fp)
   {
      fp = fopen(prefix, "r");
   }
   if (!fp)
   {
      if (myid == 0)
      {
         fprintf(stderr, "Could not open dofmap file: %s(.%05d)\n", prefix, myid);
      }
      return 0;
   }

   if (fscanf(fp, "%zu", &n) != 1 || n == 0 || n > (size_t)INT_MAX)
   {
      if (myid == 0)
      {
         fprintf(stderr, "Invalid dofmap header in %s\n", prefix);
      }
      fclose(fp);
      return 0;
   }

   vals = (HYPRE_Int *)malloc(n * sizeof(HYPRE_Int));
   if (!vals)
   {
      fclose(fp);
      return 0;
   }

   for (size_t i = 0; i < n; i++)
   {
      int entry = 0;
      if (fscanf(fp, "%d", &entry) != 1)
      {
         if (myid == 0)
         {
            fprintf(stderr, "Invalid dofmap entry while reading %s (index %zu)\n",
                    prefix, i);
         }
         free(vals);
         fclose(fp);
         return 0;
      }
      vals[i] = (HYPRE_Int)entry;
   }

   fclose(fp);
   *dofmap_out     = vals;
   *local_size_out = (HYPRE_Int)n;
   return 1;
}

int
main(int argc, char **argv)
{
   int ierr = 0;
   int mpi_initialized = 0;
   int hypre_initialized = 0;
   int myid = 0;
   int num_procs = 1;

   DriverOptions opts;
   HYPRE_IJMatrix    Aij    = NULL;
   HYPRE_IJVector    bij    = NULL;
   HYPRE_IJVector    xij    = NULL;
   HYPRE_ParCSRMatrix A     = NULL;
   HYPRE_ParVector    b     = NULL;
   HYPRE_ParVector    x     = NULL;
   HYPRE_Int         *dofmap = NULL;
   HYPRE_Int          dofmap_local_size = 0;
   HYPRE_Solver       precond = NULL;
   HYPRE_Solver       gmres   = NULL;
   HYPRE_Int          num_iters = 0;
   HYPRE_Real         final_rel_res = 0.0;
   long long          local_constraints = 0;
   long long          global_constraints = 0;
   HYPRE_BigInt       global_num_rows = 0;
   HYPRE_BigInt       global_num_cols = 0;
   HYPRE_BigInt       global_num_nonzeros = 0;
   HYPRE_BigInt       ilower = 0, iupper = -1, jl = 0, ju = -1;

   snprintf(opts.matrix_prefix, sizeof(opts.matrix_prefix),
            "test/blocks-unsym-ls_00001/IJ.out.A");
   snprintf(opts.rhs_prefix, sizeof(opts.rhs_prefix),
            "test/blocks-unsym-ls_00001/IJ.out.b");
   snprintf(opts.dofmap_prefix, sizeof(opts.dofmap_prefix),
            "test/blocks-unsym-ls_00001/dofmap.out");
   opts.x0_prefix[0] = '\0';
   opts.has_x0       = 0;
   opts.max_iter     = 200;
   opts.k_dim        = 50;
   opts.print_level  = 2;
   opts.mgr_print_level = 0;
   opts.logging      = 1;
   opts.tol          = 1.0e-8;

   if (MPI_Init(&argc, &argv))
   {
      return 1;
   }
   mpi_initialized = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   {
      int parse_status = DriverParseArgs(argc, argv, &opts);
      if (parse_status == 0)
      {
         goto cleanup;
      }
      if (parse_status < 0)
      {
         ierr = 1;
         goto cleanup;
      }
   }

   HYPRE_Initialize();
   hypre_initialized = 1;

#define CHECK_HYPRE(call)                                                     \
   do                                                                         \
   {                                                                          \
      ierr = (call);                                                          \
      if (ierr)                                                               \
      {                                                                       \
         if (myid == 0)                                                       \
         {                                                                    \
            fprintf(stderr, "hypre error %d at %s:%d in %s\n", ierr, __FILE__, \
                    __LINE__, #call);                                         \
         }                                                                    \
         goto cleanup;                                                        \
      }                                                                       \
   } while (0)

   CHECK_HYPRE(
      HYPRE_IJMatrixRead(opts.matrix_prefix, MPI_COMM_WORLD, HYPRE_PARCSR, &Aij)
   );
   CHECK_HYPRE(HYPRE_IJVectorRead(opts.rhs_prefix, MPI_COMM_WORLD, HYPRE_PARCSR, &bij));

   CHECK_HYPRE(HYPRE_IJMatrixGetGlobalInfo(Aij, &global_num_rows, &global_num_cols,
                                           &global_num_nonzeros));
   CHECK_HYPRE(HYPRE_IJMatrixGetObject(Aij, (void **)&A));
   CHECK_HYPRE(HYPRE_IJVectorGetObject(bij, (void **)&b));
   CHECK_HYPRE(HYPRE_ParCSRMatrixGetLocalRange(A, &ilower, &iupper, &jl, &ju));

   if (opts.has_x0)
   {
      CHECK_HYPRE(
         HYPRE_IJVectorRead(opts.x0_prefix, MPI_COMM_WORLD, HYPRE_PARCSR, &xij)
      );
      CHECK_HYPRE(HYPRE_IJVectorGetObject(xij, (void **)&x));
   }
   else
   {
      CHECK_HYPRE(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &xij));
      CHECK_HYPRE(HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR));
      CHECK_HYPRE(HYPRE_IJVectorInitialize(xij));
      CHECK_HYPRE(HYPRE_IJVectorAssemble(xij));
      CHECK_HYPRE(HYPRE_IJVectorGetObject(xij, (void **)&x));
      CHECK_HYPRE(HYPRE_ParVectorSetConstantValues(x, 0.0));
   }

   if (!ReadLocalDofmap(MPI_COMM_WORLD, opts.dofmap_prefix, &dofmap, &dofmap_local_size))
   {
      ierr = 1;
      goto cleanup;
   }
   for (HYPRE_Int i = 0; i < dofmap_local_size; i++)
   {
      if (dofmap[i] == 1)
      {
         local_constraints++;
      }
   }
   if (MPI_Allreduce(&local_constraints, &global_constraints, 1, MPI_LONG_LONG_INT,
                     MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
   {
      if (myid == 0)
      {
         fprintf(stderr, "MPI_Allreduce failed while counting constraints\n");
      }
      ierr = 1;
      goto cleanup;
   }

   PrintMatrixHeader(myid, num_procs, opts.matrix_prefix, global_num_rows,
                     global_num_cols, global_num_nonzeros, ilower, iupper, jl, ju,
                     local_constraints, global_constraints);

   CHECK_HYPRE(AMGF_MGRCreate(MPI_COMM_WORLD, &precond));
   CHECK_HYPRE(AMGF_MGRSetConstraintMask(precond, dofmap, dofmap_local_size));
   CHECK_HYPRE(AMGF_MGRSetPrintLevel(precond, opts.mgr_print_level));

   CHECK_HYPRE(HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &gmres));
   CHECK_HYPRE(HYPRE_GMRESSetMaxIter(gmres, opts.max_iter));
   CHECK_HYPRE(HYPRE_GMRESSetKDim(gmres, opts.k_dim));
   CHECK_HYPRE(HYPRE_GMRESSetTol(gmres, opts.tol));
   CHECK_HYPRE(HYPRE_GMRESSetPrintLevel(gmres, opts.print_level));
   CHECK_HYPRE(HYPRE_GMRESSetLogging(gmres, opts.logging));
   CHECK_HYPRE(HYPRE_ParCSRGMRESSetPrecond(gmres,
                                           (HYPRE_PtrToParSolverFcn)AMGF_MGRSolve,
                                           (HYPRE_PtrToParSolverFcn)AMGF_MGRSetup,
                                           precond));

   CHECK_HYPRE(HYPRE_ParCSRGMRESSetup(gmres, A, b, x));
   CHECK_HYPRE(HYPRE_ParCSRGMRESSolve(gmres, A, b, x));
   CHECK_HYPRE(HYPRE_GMRESGetNumIterations(gmres, &num_iters));
   CHECK_HYPRE(HYPRE_GMRESGetFinalRelativeResidualNorm(gmres, &final_rel_res));

   if (myid == 0)
   {
      printf("GMRES converged.\n");
      printf("  Iterations: %d\n", num_iters);
      printf("  Final relative residual norm: %.16e\n", final_rel_res);
   }

cleanup:
   if (gmres)
   {
      HYPRE_ParCSRGMRESDestroy(gmres);
      gmres = NULL;
   }
   if (precond)
   {
      AMGF_MGRDestroy(precond);
      precond = NULL;
   }

   free(dofmap);
   dofmap = NULL;

   if (xij)
   {
      HYPRE_IJVectorDestroy(xij);
      xij = NULL;
   }
   if (bij)
   {
      HYPRE_IJVectorDestroy(bij);
      bij = NULL;
   }
   if (Aij)
   {
      HYPRE_IJMatrixDestroy(Aij);
      Aij = NULL;
   }

   if (hypre_initialized)
   {
      HYPRE_Finalize();
   }
   if (mpi_initialized)
   {
      MPI_Finalize();
   }

   return ierr ? 1 : 0;
}
