import os,sys
sys.path.append(os.path.abspath('../data_structures'))
from data_structures.inhouse_formats import supercell

def profess_get_cell(flines,_file_name):
    """
    return the cell[i,j] : cell[i,j] is the jth cartesian component of the ith vector

    For .profess files

    Returns:
        - Final LATTICE VECTORS output to file
    Units:
        - Angstrom
    """
    cell = []

    for i,_l in enumerate(flines):
        if 'LATTICE VECTORS' in _l:
            # check units
            assert _l.split()[2].lower()=='(angstrom)','{} units are not supported'.\
            format(_l.split()[2].split('(')[0].split(')')[0])

            cell += [  [[float(v) for v in flines[i+1].split()[2:]],\
                       [float(v) for v in flines[i+2].split()[2:]],\
                       [float(v) for v in flines[i+3].split()[2:]]]   ]
    
    if len(cell)!=0:
        cell = cell[-1]
    
    return cell

def ion_get_cell(flines,_file_name):
    """
    return cell[i,j] as jth cartesian component of ith cell vector
    for .ion and .final.geom files
    """
    import numpy as np

    cart_break = []

    for i,_l in enumerate(flines):
        if '%block lattice_cart' in _l.lower():
            cart_break.append(i)
        elif '%endblock lattice_cart' in _l.lower() or '%end block lattice_cart' in _l.lower():
            cart_break.append(i)

    assert len(cart_break)==2,'error parsing cell file {}'.format(_file_name)

    cell = []
    for _l in flines[cart_break[0]+1:cart_break[1]]:
        if len(_l.split())==3:
            cell += [ [float(_v) for _v in _l.split()] ]

    assert len(cell)==3,'error parsing cell file {}'.format(_file_name)

    return np.asarray(cell)


def read_profess_profess(fd,verbose=False):
    """
    Input:
        - fd : an open file object
        
    Output:
        - list of supercell objects
    """
    import numpy as np

    def _get_final_energy(flines):
        """
        return the final total energy in flines
        
        Units:
            - eV
        """
        energy = []
        
        for i,_l in enumerate(flines):
            if 'FINAL ENERGIES' in _l:
                energy += [float(flines[i+13].split()[4])]
                
                # check units
                assert flines[i+13].split()[5].lower()=='ev','{} units are not supported'.\
                format(flines[i+13].split()[5])
        
        assert len(energy)==1,'error parsing final energy of profess file {}'.format(fd.name)

        return energy[0]

    def _get_stress(flines):
        """
        return stress matrix or None if no stress calculate was made

        Returns:
            - Final STRESS (GPa) block found in file, or None

        Units:
            - GPa
        """
        stress = []

        # fetch Stress in GPa
        for i,_l in enumerate(flines):
            if 'STRESS (GPa)' in _l:
                
                stress += [  [[float(v) for v in flines[i+1].split()[2:]],\
                             [float(v) for v in flines[i+2].split()[2:]],\
                             [float(v) for v in flines[i+3].split()[2:]]]   ]

        if len(stress)==0:
            # no stress calculation was made
            stress = [None]

        return stress[-1]

    def _get_forces(flines):
        """
        return atom forces, or None if not present
        
        Units:
            - eV/A
        """
        import numpy as np

        forces = []

        for i,_l in enumerate(flines):
            if 'TOTAL-FORCE (eV/A)' in _l:
                for j,_m in enumerate(flines[i+3:]):
                    if '................' in _m:
                        block_break = j
                        break
                 
                for j,_m in enumerate(flines[i+3:i+3+block_break]):
                    forces += [ [float(_v) for _v in _m.split()[3:-1] ] ]
        
        if len(forces)==0:
            # no forces were output
            forces = None

        return np.asarray(forces)

    def _get_species(flines):
        species = []

        for i,_l in enumerate(flines):
            if 'TOTAL-FORCE (eV/A)' in _l:
                for j,_m in enumerate(flines[i+3:]):
                    if '................' in _m:
                        block_break = j
                        break
                
                for j,_m in enumerate(flines[i+3:i+3+block_break]):
                    species += [_m.split()[1]]
        
        if len(species)==0:
            # no forces were output
            species = None
        
        return species 

    flines = fd.readlines()

    #-------#
    # parse #
    #-------#

    energy = _get_final_energy(flines)

    cell = profess_get_cell(flines,fd.name)

    stress = _get_stress(flines)

    forces = _get_forces(flines)

    species = _get_species(flines)

    #-----------------------#
    # instantiate supercell #
    #-----------------------#

    structure = supercell()

    # energy 0K
    structure["energy"] = energy

    # cell vectors
    if len(cell)!=0:
        structure["cell"] = cell

    # stress
    if np.size(stress)!=1:
        structure["stress"] = stress
    
    # forces
    if np.size(forces)!=1:
        structure["forces"] = forces

    # atom species
    if isinstance(species,type(None)) is not True:
        structure["species"] = species

    # file name
    structure["files"] = [fd.name.split('/')[-1]]

    # use prefix for name
    structure["name"] = fd.name.split('/')[-1].split('.')[0]

    return [structure]


def read_profess_ion(fd,verbose=False):
    """
    return a list containing a single structure object from the castep cell file format
    """
    import numpy as np

    def _safety_checks(flines,_file_name):
        """
        check for cell vectors and atom positions
        """

        # check for cell vector keywords
        assert (any(['%block lattice_cart' in _l.lower() for _l in flines])) and \
        (any(['%endblock lattice_cart' in _l.lower() or '%end block lattice_cart' \
        in _l.lower() for _l in flines])),\
        'error parsing cell vectors in {}'.format(_file_name)

        # check for positions_frac keywords
        assert ( (any(['%block positions_frac' in _l.lower() for _l in flines])) and \
        (any(['%endblock positions_frac' in _l.lower() or '%end block positions_frac' \
        in _l.lower() for _l in flines])) ) or \
               ( (any(['%block positions_cart' in _l.lower() for _l in flines])) and \
        (any(['%endblock positions_cart' in _l.lower() or '%end block positions_cart' \
        in _l.lower() for _l in flines])) ),\
        'error parsing atom positions in {}'.format(_file_name)

    def _get_positions(flines,_file_name,cell):
        """
        return fractional coordinates of atoms wrt to cell vectors
        """
        def _convert_cart_to_frac(fracpos,cell):
            invcell = np.linalg.inv(cell)

            for i in range(len(fracpos)):
                fracpos[i] = [sum([fracpos[i][j]*invcell[j][k] for j in range(3)]) for k in range(3)]

            return np.array(fracpos)


        pos_break = []

        have_cartesians = False

        for i,_l in enumerate(flines):
            if '%block positions_frac' in _l.lower():
                pos_break.append(i)
            elif '%endblock positions_frac' in _l.lower() or '%end block positions_frac' in _l.lower():
                pos_break.append(i)
            elif '%block positions_cart' in _l.lower():
                pos_break.append(i)
                have_cartesians = True
            elif '%endblock positions_cart' in _l.lower() or '%end block positions_cart' in _l.lower():
                pos_break.append(i)
            

        assert len(pos_break)==2,'error parsing cell file {}'.format(_file_name)

        fracpos = []

        for _l in flines[pos_break[0]+1:pos_break[1]]:
            if len(_l.split())==4:
                fracpos += [ [float(_v) for _v in _l.split()[1:]] ]

        if have_cartesians:
            fracpos = _convert_cart_to_frac(fracpos,cell)

        return np.asarray(fracpos)

    def _get_species(flines,_file_name):
        """
        return species list for cell file
        """
        
        pos_break = []

        for i,_l in enumerate(flines):
            if '%block positions_frac' in _l.lower() or '%block positions_cart' in _l.lower():
                pos_break.append(i)
            elif '%endblock positions_frac' in _l.lower() or '%end block positions_frac' in _l.lower() or\
                 '%endblock positions_cart' in _l.lower() or '%end block positions_cart' in _l.lower():
                 pos_break.append(i)

        assert len(pos_break)==2,'error parsing cell file {}'.format(_file_name)

        species = []

        for _l in flines[pos_break[0]+1:pos_break[1]]:
            if len(_l.split())==4:
                species.append(_l.split()[0])

        return species

    # read lines
    flines = fd.readlines()
    
    # check file for basic keywords
    _safety_checks(flines,fd.name)

    #-------#
    # parse #
    #-------#

    cell = ion_get_cell(flines,fd.name)

    fracpos = _get_positions(flines,fd.name,cell)

    species = _get_species(flines,fd.name)
    
    #-----------------------#
    # instantiate supercell #
    #-----------------------#

    structure = supercell()

    # cell vectors
    structure["cell"] = cell

    # fractional coordinates of atoms
    structure["positions"] = fracpos

    # atom species list
    structure["species"] = species
    
    # file name
    structure["files"] = [fd.name.split('/')[-1]]

    # use prefix for name
    structure["name"] = fd.name.split('/')[-1].split('.')[0]

    return [structure]

def read_profess_den(fd,verbose=False):
    """
    return the electronic density per unit volume as a function of fractional coordinates

    4th element is the number of spins

    Units:
        eV / (A^3)
    """
    import numpy as np

    def _safety_checks(fd,grid_size,lengthdata):
        """
        check for easy to notice corruption
        """
        assert lengthdata==grid_size[0]*grid_size[1]*grid_size[2],\
                '{} is thought to be corrupted {} != {} '.format(fd.name,lengthdata,grid_size[0]*grid_size[1]*grid_size[2])

    def _grid_size(flines):
        """
        return the number of points in each lattice vector direction

        Returns:
            - [m1G,m2G,m3G]
        """
        
        return [int(flines[0][:200].split()[1+2*i]) for i in range(3)]+[int(flines[0][:200].split()[9])] 


    def _grid_values(flines):
        """
        return raw grid values in 1-d array
        """

        firstentry = flines[0][:200].split()[10]


        for i in range(len(firstentry),200):
            if firstentry == flines[0][i-len(firstentry):i]:
                headerend = i - len(firstentry)
                break
        return np.array([float(_v) for _v in flines[0].split()[10:]])
        #return np.array([float(_v) for _v in flines[0][headerend:].split()])

    def _array_to_dict(grid_size,raw_data,cell):
        """
        scale densities to e/(A^3) rather than e/(Bohr^3) and create dict entries for each point

        assume u=[0,1-1/m1G],v=[0,1-1/m2G],w=[0,1-1/m3G],s=[0,grid_size[3]-1] correspond to 
        raw_data[ s*(m1G*m2G*m3G) + w*(m1G*m2G) + v*m1G + u ]

        """
        def _fractocart(gamma,cell):
            """
            return cartesian from fractional coordinates and cell vectors

            Input:
                - cell[i,j] is the jth cartesian component of the ith cell vector
                - gamma[i] is the fractional coordinate wrt. the ith cell vector

            Output:
                -r[i] = sum_{j} gamma[j]*cell[j][i]
            """
            
            r = [0.0,0.0,0.0]

            for k in range(3):
                r[k] = sum([gamma[j]*cell[j][k] for j in range(3)])
            
            return np.asarray(r)

        m1G,m2G,m3G = grid_size[:3]
        
        points = np.zeros((len(raw_data),3),dtype=float)

        cntr = 0

        for z in range(grid_size[3]):
            for k in range(m3G):
                for j in range(m2G):
                    for i in range(m1G):
                        # scale to fractional coordinates assuming _u = [0,1-1/m1G]
                       _u = i/m1G
                       _v = j/m2G
                       _w = k/m3G

                       points[cntr] = _fractocart([_u,_v,_w],cell)

                       cntr += 1

        # scale raw density from 1/(Bohr^3) to 1/(A^3)
        conversion_factor = 1.8897259886**3 
        
        raw_data *= conversion_factor

        """ deprecated
        _s = np.floor(i/(m1G*m2G*m3G))

        _w = np.floor( (i-(m1G*m2G*m3G)*_s) / (m1G*m2G) )

        _v = np.floor( (i-(m1G*m2G*m3G)*_s-(m1G*m2G)*_w) / (m1G) )

        _u = i - (m1G*m2G*m3G)*_s - (m1G*m2G)*_w - (m1G)*_v"""

        return {"xyz":points,"density":raw_data}

    def _get_cell_from_file(fd):
        """
        search the directory of the open file object, fd, for .ion, .final.geom or .profess
        files (which share the same prefix as fd), from which the cell vectors can be parsed
        """
        
        import os

        file_path = '/'.join(fd.name.split('/')[:-1])
        if len(file_path)==0:
            file_path ='.'
        files = os.listdir(file_path)
        prefix = fd.name.split('/')[-1].split('.')[0]
        present_files = {}


        for _f in files:
            # search for which files with same prefix are present
            if _f.split('.')[0]==prefix and _f.split('.')[-1].lower()=='profess':
                present_files.update({'profess':file_path+'/'+_f})
            elif _f.split('.')[0]==prefix and _f.split('.')[-1].lower()=='ion':
                present_files.update({'ion':file_path+'/'+_f})
            elif _f.split('.')[0]==prefix and '.'.join(_f.split('.')[1:]).lower()=='final.geom':
                present_files.update({'final.geom':file_path+'/'+_f})

        assert len(present_files)!=0,'profess density file {} must be accompanied by a .profess, .ion, .final.geom file with the same prefix, in the same directory. Please move one of these files here: {}'.\
        format(fd.name,file_path)

        if 'profess' in present_files:
            with open(present_files["profess"],'r') as f:
                cell = profess_get_cell(f.readlines(),f.name)

                if len(cell)==0:
                    if 'final.geom' in present_files:
                        with open(present_files["final.geom"],'r') as f:
                            cell = ion_get_cell(f.readlines(),f.name)
                    elif 'ion' in present_files:
                        with open(present_files["ion"],'r') as f:
                            cell = ion_get_cell(f.readlines(),f.name)
                    
        elif 'final.geom' in present_files:
            with open(present_files["final.geom"],'r') as f:
                cell = ion_get_cell(f.readlines(),f.name)
        elif 'ion' in present_files:
            with open(present_files["ion"],'r') as f:
                cell = ion_get_cell(f.readlines(),f.name)

        assert np.shape(cell)==(3,3),'error parsing cell for {}'.format(fd.name)
    
        return cell

    with open(fd.name) as f:
        flines = f.readlines()

    #------------#
    # parse data #
    #------------#
    
    # must attempt to fetch cell vectors from another file in the same directory as fd
    cell = _get_cell_from_file(fd)

    # get number of points along each cell vector
    grid_size = _grid_size(flines)

    # unfortunately, spin!=1 is not supported yet
    assert grid_size[3]==1,'{} many spins is not supported yet, only 1 many spins are so far'.format(grid_size[3])

    # grid values in a 1-d array
    raw_data = _grid_values(flines)

    # check for corrupted file
    _safety_checks(fd,grid_size,len(raw_data))

    # scale to e/(A^3) and create density dict entries
    den_dict = _array_to_dict(grid_size,raw_data,cell)

    structure = supercell()

    structure["cell"] = cell

    # use prefix for name
    structure["name"] = fd.name.split('/')[-1].split('.')[0]

    structure["edensity"] = den_dict

    structure["files"] = [fd.name.split('/')[-1]]

    return [structure]

class parse:
    """
    Class to interface profess parsing with general parsing (end user) class

    Supported file types are self.implemented_file_types

    MOLECULAR DYNAMICS CALCULATIONS ARE NOT YET SUPPORTED

    Example
    -------
    # initialise instance
    profess_parser = parse('/path-to-file/<file_name>.<file_type>,file_type=<file_type>)

    # parse
    "/path-to-file/<file_name>.<file_type>" : "profess_parser.run()

    # extract supercell
    supercell = profess_parser.get_supercells()
    """

    def __init__(self,path,file_type):
        self.implemented_file_types = ['profess','ion','final.geom','den']

        assert file_type in self.implemented_file_types,'Assertion failed - got unexpected file_type {}. Expected on of: {}'.format(file_type,self.implemented_file_types)

        self.file_type = file_type  # file type
        self.path = path            # full path to file
        self.supercells = None      # list of supercell objects

        self.internal_interface = {'profess':read_profess_profess,'ion':read_profess_ion,\
        'final.geom':read_profess_ion,'den':read_profess_den}

    def run(self,verbose=False):
        """
        method to run parser once instanciated
        """
        with open(self.path,'r') as f:
            self.supercells = self.internal_interface[self.file_type](f,verbose=verbose)

    def get_supercells(self):
        """
        method to return supercells object associated with this parser instance
        """
        return self.supercells

class wrap_inhouse:
    """
    wrap inhouse supercell class and provide interface to profess i/o
    """
    
    def __init__(self,structure,seedname=None):
        import copy
        
        assert isinstance(structure,supercell),'structure must be a supercell object'

        self.supercell = copy.deepcopy(structure)

        # prefix for profess input files
        if seedname is None:
            if structure["name"] is not None:
                self.name = structure["name"]
            else:
                self.name = "unnamed"
        else:
            self.name = seedname

        if len(self.name.split('.'))!=0:
            # strip file name if necessary to retain prefix only
            tmp = self.name.split('.')
            if tmp[-1].lower()=='profess' or tmp[-1].lower()=='ion' or tmp[-1].lower()=='den':
                self.name = '.'.join(self.name.split('.')[:-1])

    def write_ion(self,seedname=None):
        """
        write a profess(castep) ion(cell) file for atom positions and cell vectors
        """
        import os

        if seedname is None:
            if self.name is not None:
                seedname = self.name
            else:
                seedname = "unnamed"


        assert self.supercell["positions"] is not None,'{}.ion file needs positions'.format(seedname)
        assert self.supercell["cell"] is not None,'{}.ion file needs cell vectors'.format(seedname)
        assert self.supercell["species"] is not None,'{}.ion file needs atom species'.format(seedname)

        flines = ['%BLOCK LATTICE_CART\n']
        for _l in self.supercell["cell"]:
            flines.append('{:<20f} {:<20f} {:<20f}\n'.format(_l[0],_l[1],_l[2]))
        flines.append('%ENDBLOCK LATTICE_CART\n\n')

        flines.append('%BLOCK POSITIONS_FRAC\n')
        for i,_pos in enumerate(self.supercell["positions"]):
            flines.append('{:<5} {:<20f} {:<20f} {:<20f}\n'.format(self.supercell["species"][i],\
            _pos[0],_pos[1],_pos[2]))
        flines.append('%ENDBLOCK POSITIONS_FRAC\n\n')

        if any(_s.lower() == 'al' for _s in self.supercell["species"]):
            flines.append('%BLOCK SPECIES_POT\n')
            flines.append('Al al_HC.lda.recpot\n')
            flines.append('%ENDBLOCK SPECIES_POT\n')

        with open(seedname+'.ion','w') as f:
            f.writelines(flines)

    def write_inpt(self,seedname=None,ecut=600,method='non',kinetic='wt',xc='lda',stress=True,\
                   forces=True,testgrid=False):
        """
        write profess parameters input, .inpt file
        """
        if seedname is None:
            if self.name is not None:
                seedname = self.name
            else:
                seedname = "unnamed"


        flines = ['ecut         {}\n'.format(ecut),
                  'method       {}\n'.format(method),
                  'kinetic      {}\n'.format(kinetic),
                  'exch         {}\n'.format(xc)]
        if method=='non':
            flines.append('rhof         {}\n'.format(seedname+'.den'))
        if stress:
            flines.append('calculate stresses\n')
        if forces:
            flines.append('calculate forces\n')
        if testgrid:
            # print out expected grid size
            flines.append('#query\n')

        with open(seedname+'.inpt','w') as f:
            f.writelines(flines)

    def write_den(self,xyz=None,density=None,seedname=None,zero_negatives=True):
        """
        0. calculate m1G,m2G,m3G, number of grid points per dimension
        1. convert xyz (cartesian) to uvw (fractional coordinates)
        2. convert uvw to _u_v_w (grid points)
        3. _u = u/m1G = [0,m1G-1] , _v = v/m2G = [0,m2G-1] , _w = w/m3G = [0,m3G-1]
        """

        import numpy as np
        import time

        def _fraccoords(xyz):
            invcell = np.linalg.inv(self.supercell["cell"])
            
            tmp = np.zeros((len(xyz),3),dtype=float,order='C')

            for i,_r in enumerate(xyz):
                tmp[i] = _carttofrac(_r,invcell)

            return tmp

        def _carttofrac(dr,invcell):
            df = [0,0,0]

            for i in range(3):
                df[i] = sum([dr[j]*invcell[j,i] for j in range(3)])

            return df

        def _calcgridnumber(frac):
            """
            return number of grid points per dimension [m1G,m2G,m3G]
            """
            
            # round to precision many decimals in fractional coordinates
            precision = 8

            numpoints = [set(np.round(frac[:,0],decimals=precision)),\
                         set(np.round(frac[:,1],decimals=precision)),\
                         set(np.round(frac[:,2],decimals=precision))]
            
            interval = [set([]),set([]),set([])]

            for i in range(3):
                tmp = sorted(list(numpoints[i]))

                for j,_a in enumerate(tmp):
                    if j < len(numpoints[i])-1:
                        interval[i].update(set([tmp[j+1]-tmp[j]]))


            for i in range(3):
                delete_me = []
                for _a in interval[i]:
                    if np.isclose(_a,0.0,atol=1E-9):
                        delete_me.append(_a)
                for _a in delete_me:
                    interval[i].remove(_a)
            
            # method 1
            numpoints = [len(n) for n in numpoints]
            
            # method 2
            grid = [int(round(1.0/sorted(list(interval[i]))[0])) for i in range(3)]
            
            assert all([grid[ii]==numpoints[ii] for ii in range(3)]),\
                    "disagreement between both methods to calculation number of grid points. Check this section"
            assert grid[0]*grid[1]*grid[2]==len(frac),'error with m1G={} m2G={} m3G={}. Check grid is complete'.\
            format(grid[0],grid[1],grid[2])

            return grid

        def _fractogrid(frac,mGs,density):
            """
            convert fractional to grid coordinates
            
            grid number = frac cord / mG  = [0,mG-1]
            """
            
            def _index(mGs,gridpoint):
                """
                gridpoint[i] = [0,mGs[i]-1]
                
                index[gridpoint] = gridpoint[0] + gridpoint[1]*mGs[0] + gridpoint[2]*mGs[0]*mGs[1]
                """
                return gridpoint[0]+gridpoint[1]*mGs[0]+gridpoint[2]*mGs[0]*mGs[1]
            
            grid = np.zeros((len(frac),3),dtype=int,order='C')

            for i,_fpos in enumerate(frac):
                tmp = [_fpos[j]*mGs[j] for j in range(3)]

                for j in range(3):
                    grid[i,j] = round(tmp[j])

                assert np.allclose(grid[i,1:3],tmp[1:3],atol=1e-8),'density grid is not homogenous'
    
                for j in range(3):
                    if np.isclose(_fpos[j],1.0,atol=1e-8):
                        grid[i,j] = 0

            # check for duplicate grid points
            tmp = [tuple(_point) for _point in grid]

            assert len(set(tmp))==len(frac),'holes or duplicate points found in density grid'
           
            # combine grid point and density into a dict to sort
            griddict = [{"grid":grid[i],"density":density[i]} for i in range(len(density))]

            # order grid points by order of appearance in .den file
            return sorted(griddict,key = lambda _point : _index(mGs,_point["grid"]))

        
        
        if density is None or xyz is None:
            density = self.supercell["edensity"]["density"]
            
            xyz = self.supercell["edensity"]["xyz"]
        if seedname is None:
            if self.name is not None:
                seedname = self.name
            else:
                seedname = "unnamed"
        

        # fractional coordinates
        frac = _fraccoords(xyz)

        # number of grid points
        mGs = _calcgridnumber(frac)

        t1 = time.time()

        # grid points and densities in order of appearance in .den file
        griddictionary = _fractogrid(frac,mGs,density)
        
        t2 = time.time()

        print ('preparing and checking density grid for output to profess {} s...'.format(t2-t1))

        # conversion from 1/(Ang^3) to 1/(Bohr^3)
        AngToBohr = 1.0/(1.8897259885789**3)

        newflines = ['  x-dimension:{:12}   y-dimension:{:12}   z-dimension:{:12}   # of spins: {:12}'\
        .format(mGs[0],mGs[1],mGs[2],1)]

        for _point in griddictionary:
            if zero_negatives and _point["density"] < 0.0:
                # ZERO NEGATIVE DENSITIES WHEN WRITING TO FILE
                newflines.append(' {:>26.16e}'.format(0.0))
            else:
                newflines.append(' {:>26.16e}'.format(_point["density"]*AngToBohr))

        with open(seedname+'.den','w') as f:
            f.writelines(newflines)

    def recommended_gridsize(self,ecut=600):
        """
        return size of profess density grid size recommended for given ecut and supercell
        
        ASSUMES modified PROFESS3 binary is in path list as "profess"
        """
        import os
        import numpy as np

        # temporary out file
        tmpfile1 = 'tmp'
        tmpfile2 = 'tmp.out_grid'

        # write .ion and .inpt files
        self.write_ion(seedname=tmpfile1)
        self.write_inpt(seedname=tmpfile1,ecut=ecut,method='ntn',testgrid=True)

        # run profess 
        os.system('profess '+tmpfile1+' > '+tmpfile2)

        m1G = None
        m2G = None
        m3G = None
        numspin = None

        # read recommended grid
        with open(tmpfile2,'r') as f:
            flines = f.readlines()

            for _l in flines:
                if 'cell vector 1' in _l:
                    m1G = int(_l.split()[4])
                elif 'cell vector 2' in _l:
                    m2G = int(_l.split()[4])
                elif 'cell vector 3' in _l:
                    m3G = int(_l.split()[4])
                elif 'spin states per grid point' in _l:
                    numspin = int(_l.split()[6])
        
        assert m1G is not None and m2G is not None and m3G is not None and numspin is not None,\
        'error with profess run'
        assert numspin==1,'only spin polarized (# spin states = 1) calculations are supported'

        # clean up scratch files
        os.remove(tmpfile1+'.ion')
        os.remove(tmpfile1+'.inpt')
        os.remove(tmpfile1+'.out')
        os.remove(tmpfile1+'.err')
        os.remove(tmpfile2)
        
        return np.array([m1G,m2G,m3G,numspin],dtype=int)

    def write_singlepoint(self,seedname=None,xyz=None,density=None,ecut=600,kinetic='wt',xc='lda'):
        """
        Fractional coordinates | species | cell vectors | e density
        ------------------------------------------------------------------
             self.supercell    ->  ""    ->    ""       | method arguments
        
        take fractional coordinates, species type and cell vectors from
        self.supercell. 

        If xyz==None or density==None, take electron density from self.supercell.
        Else, take from argument list of method and IGNORE electron density in 
        self.supercell if present.

        Output
        ------
            - self.supercell.name+'.ion'
            - self.supercell.name+'.inpt'
            - self.supercell.name+'.den'
        """

        def _checks(xyz,density,ecut,kinetic,xc):
            assert self.supercell["positions"] is not None,'fractional positions must be set in {}'.\
            format(self.name)
            assert self.supercell["cell"] is not None,'cell vectors must be set in {}'.format(self.name)
            assert self.supercell["species"] is not None,'atom types must be set in {}'.format(self.name)
            if xyz is None or density is None:
                assert self.supercell["edensity"] is not None,'e density in {} must be set'.format(self.name)
            assert isinstance(ecut,(float,int)),'ecut={} is not a number'.format(ecut)
            assert kinetic.lower() in ['wt','tho','tf','von','vw','wtv','wgc','wgv','lq','hq','cat','cav','hc',
                                       'dec','dev','evt','evc','gga','vw'],'{} is not a recnognised KE form'.\
                                       format(kinetic)
            assert xc.lower() in ['lda','pbe'],'{} is not a recognised xc functional'.format(xc)

        _checks(xyz,density,ecut,kinetic,xc)

        if seedname is None:
            seedname = self.name


        # output atom positions and cell vectors
        self.write_ion(seedname=seedname)

        # write calculation details
        self.write_inpt(seedname,ecut,'non',kinetic,xc,True,True,False)

        # write electron density
        self.write_den(xyz=xyz,density=density,seedname=seedname)
