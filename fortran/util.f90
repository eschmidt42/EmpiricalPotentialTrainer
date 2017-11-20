!==================================================================!
! NOTES                                                            !
! -----                                                            !
!   - for OpenMP and gnu, compile with --f90flags="-fopenmp -lgomp"!
!==================================================================!

module util
    implicit none

    contains
        integer function get_num_threads()
            !============================================================!
            ! Return the number of threads used in parallelised sections !
            ! of fortran, for given OMP_NUM_THREADS in unix environment  !
            !                                                            !
            ! To change this, export OMP_NUM_THREADS = <x> before        !
            ! executing Python                                           !
            !============================================================!

            !$ use omp_lib

            implicit none

            get_num_threads = omp_get_max_threads()
        
        end function get_num_threads


end module util
