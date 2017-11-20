#!/bin/bash

#--------------------------------------------------------------#
# Make script for fortran modules wrapped by Python interfaces #
#                                                              #
# See README for requirements and install guide                #
#--------------------------------------------------------------#

# fortran compiler
FC="gfortran"

# f2py binary name
f2py="f2py"

# f90wrap binary name
f90wrap="f90wrap"

modname="assorted"

filename1="bond_generation.f90"
filename2="rvm_basis.f90"
filename3="util.f90"

$FC -c -W -Wall -pedantic $filename1 -fPIC -O2
$FC -c -W -Wall -pedantic $filename2 -fPIC -O2 -fopenmp -lgomp
$FC -c -W -Wall -pedantic $filename3 -fPIC -O2 -fopenmp -lgomp

$f90wrap -m $modname $filename1 $filename2 $filename3 -k kind_map -S 12 

# include OpenMP routines
$f2py -c -m $modname f90wrap_*.f90 *.o --f90flags="-fPIC -fopenmp" -lgomp --fcompiler=$FC 
