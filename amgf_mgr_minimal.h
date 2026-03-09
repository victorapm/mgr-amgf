#ifndef AMGF_MGR_MINIMAL_H
#define AMGF_MGR_MINIMAL_H

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

HYPRE_Int AMGF_MGRCreate(MPI_Comm comm, HYPRE_Solver *solver);
HYPRE_Int AMGF_MGRSetConstraintMask(HYPRE_Solver solver,
                                    const HYPRE_Int *constraint_mask,
                                    HYPRE_Int local_num_rows);
HYPRE_Int AMGF_MGRSetPrintLevel(HYPRE_Solver solver, HYPRE_Int print_level);
HYPRE_Int AMGF_MGRSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b, HYPRE_ParVector x);
HYPRE_Int AMGF_MGRSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b, HYPRE_ParVector x);
HYPRE_Int AMGF_MGRDestroy(HYPRE_Solver solver);

#endif
