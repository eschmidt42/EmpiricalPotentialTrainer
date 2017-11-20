module bond_generation
    implicit none

    contains
        subroutine meam_bond_generator(r_el1,r_el2,r_vec_el1,r_vec_el2,Nel1,Nel2,num_pairs,same_el,ani_out)
            !===========================================================!
            ! calculate MEAM anisotropic bond contribution for all      !
            ! el1,el2 atom pairs                                        !
            !===========================================================!

            implicit none

            !* arguments
            logical,intent(in) :: same_el
            integer,intent(in) :: Nel1,Nel2,num_pairs
            real(8),intent(in) :: r_el1(1:Nel1),r_el2(1:Nel2)
            real(8),intent(in) :: r_vec_el1(1:3,1:Nel1),r_vec_el2(1:3,1:Nel2)
            real(8),intent(out) :: ani_out(1:3,1:num_pairs)
        
            !* scratch
            integer :: ii,jj,idx,start_idx,cntr
            real(8) :: tmp

            cntr = 1

            do ii=1,Nel1,1
                
                !* calculate only unique pairs
                if (same_el) then 
                    start_idx = ii+1
                else
                    start_idx = 1
                end if 
                
                do jj=start_idx,Nel2,1
                    idx = cntr

                    ani_out(1,idx) = r_el1(ii)
                    ani_out(2,idx) = r_el2(jj)
                    !* assume r_vec_el1(2) are unit vectors here! take arccos
                    tmp = r_vec_el1(1,ii)*r_vec_el2(1,jj)+r_vec_el1(2,ii)*r_vec_el2(2,jj)+&
                    &r_vec_el1(3,ii)*r_vec_el2(3,jj)
                  
                    
                    if (tmp>1.0d0) then
                        !* assume any error is just floating point! :/
                        tmp = 1.0d0
                    else if (tmp < -1.0d0) then
                        tmp = -1.0d0
                    end if
                    
                    ani_out(3,idx) = dacos(tmp)
                
                    cntr = cntr + 1
                end do
            end do
        end subroutine MEAM_bond_generator
        
        subroutine query_ball_point(atoms,Natm,grid,Ngrid,rcut,atom_idxs,num_neighbours)
            implicit none

            integer,intent(in) :: Natm,Ngrid
            real(8),intent(in) :: atoms(1:3,Natm),grid(1:3,1:Ngrid),rcut
            integer,intent(out) :: atom_idxs(1:Natm,1:Ngrid),num_neighbours(1:Ngrid)

            real(8) :: rcut2,grid_point(1:3)
            integer :: ii,jj

            rcut2 = rcut**2

            do ii=1,Ngrid,1
                grid_point(1:3) = grid(1:3,ii)

                num_neighbours(ii) = 0

                do jj=1,Natm,1
                    if ( (atoms(1,jj)-grid_point(1))**2+(atoms(2,jj)-grid_point(2))**2+&
                    &(atoms(3,jj)-grid_point(3))**2 .le.rcut2 ) then
                        !* remember to offset indices to zero for python!!
                        atom_idxs(num_neighbours(ii)+1,ii) = jj - 1

                        num_neighbours(ii) = num_neighbours(ii) + 1
                    end if
                end do
            end do

        end subroutine query_ball_point

end module bond_generation
