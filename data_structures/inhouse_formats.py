#!/bin/env python3
"""
module for in-house classes for data format and storage

Contents:
    - supercell() : atomic supercell information

es611@cam.ac.uk , atf29@cam.ac.uk  
"""

import numpy as np
import copy, warnings

def deprecate(fun):
    warnings.warn("This function ({} > {}) will be removed in future updates!".format(fun.__module__,fun.__name__),DeprecationWarning)
    return fun

def assertion_statement(expected,varname,got):
    return "Assertion failed - expected '{}' to be one of: {}. Got {} instead.".format(varname,expected,got)

class supercell:
    """In-house data structure for atomic supercells.

    attributes in square brakets [] are optionaly set, those that are not are mandatory.
   
    Parameters
    ----------
    fast : boolean
        Deactivates the safety tests for setting attributes. 
   
    Attributes
    ----------
        - 1.  cell           : cartesian supercell vectors / (A)
        - 2.  positions      : fractional atom coordinates
        - 3.  [ species      : atom types                 ]  
        - 4.  [ energy       : total energy / (eV)        ]
        - 5.  [ forces       : atomic forces / (eV/A)     ]
        - 6.  [ stress       : pressure tensor / (GPa)    ]
        - 7.  [ charge       : electronic charge / (|e-|) ]
        - 8.  name : structure name
        - 9.  [ bulkmodulus : bulk modulus / (GPa)        ]
        - 10. [ spacegroup   : space group of cyrstal     ]
        - 11. [ edensity     : electron density /(???)    ]
        - 12. [ files        : files contributing to structure ]
        - 13. [ enthalpy     : enthalpy H = U + PV /(eV)  ]

    Notes
    -----
    Recomended use of setting attributes is directly via 
    the __call__ method where as key the exact attribute name is expected
    or as with a dict. To set values each attribute has a designated set_method.
    For retrieving info one can access the attributes via dict syntax
    or respective get_method.
    
    Important: When setting the name of the supercell make sure its prefix 
    is unique when reading multiple files such that each complete structure
    can be constructed from all supercells with identical prefix and varying suffix. 
    """

    def __init__(self,fast=False,ase_atoms=None):
        
        self.properties = ["cell","positions","species","energy","forces","stress","charge","name",\
            "bulkmodulus","spacegroup","edensity","files","enthalpy"]

        # method names to set class attributes
        self.set_methods = {"cell":"set_cell","positions":"set_positions","species":"set_species",\
            "forces":"set_forces","energy":"set_energy","stress":"set_stress","charge":"set_charge","name":"set_name",\
            "bulkmodulus":"set_bulkmodulus","spacegroup":"set_spacegroup","edensity":"set_edensity",\
            "files":"set_files","enthalpy":"set_enthalpy"}
        self.iproperties = set(self.set_methods.keys()) #implemented properties
        
        # method names to get class attributes
        self.get_methods = {"cell":"get_cell","positions":"get_positions","species":"get_species",\
            "forces":"get_forces","energy":"get_energy","stress":"get_stress","charge":"get_charge","name":"get_name",\
            "bulkmodulus":"get_bulkmodulus","spacegroup":"get_spacegroup","edensity":"get_edensity",\
            "files":"get_files","enthalpy":"get_enthalpy"}

        for _p in self.properties:
            setattr(self,_p,None)

        # implementation error messages
        self.errors = {\
        'E1':'\nThe supercell given must be: cell[ia][ib] is the ibth cartesian component \
        of the iath cell vector\n',\
        'E2':'\nThe fractional coordinates passed to supercell.set_frac() must be a 2-d \
        list or numpy array: frac[ib][ia] is the iath fractional coordinate of the ibth atom',\
        'E3':'\nThe atom names passed to self.atyp must a list of character strings',\
        'E4':'\nThe supercell total energy passed must be a float',\
        'E5':'\nThe cartesian component of force /(eV/A) passed to supercell.afrc must be \
        a 2-d list or numpy array: force[ib][ia] is the iath cartesian coordinate of force \
        /(eV/A) of the ibth atom',\
        'E6':'\nCrystal symmetry not recognised as a supported convention',\
        'E7':'\nElectron density must be formatted as a list of dictionaries. See documentation\n',\
        'E8':'\nUnique structure name is already in use on the stack\n',\
        'E9':'\nGeneral implementation error\n',\
        'E10':'\nThe supplied atom charges must all be real numbers\n',\
        'E11':'\nThe Bulk modulus supplied must be a real number\n',\
        'E12':'\nThe crystal symmetry must be supplied as a dictionary with keys number or Hermann-Mauguin\n',\
        'E13':'\nMethod set_ase_atoms accepts only an ASE Atoms object'}
        
        
        #switch to turn safety checks on or off
        self.fast = fast

        # if ase atoms object is supplied, read and set properties in supercell
        if ase_atoms is not None:
            self.set_ase_atoms(ase_atoms)

    def __del__(self):
        """
        delete an instance of this class. subroutine called upon del <object>
        """

    def __eq__(self,obj):
        return self.name == obj.name 
    def __lt__(self,obj):
        return self.name < obj.name
    def __gt__(self,obj):
        return self.name > obj.name
    def __le__(self,obj):
        return self.name <= obj.name
    def __ge__(self,obj):
        return self.name >= obj.name
    def __ne__(self,obj):
        return self.name != obj.name
    

    def clean(self):
        for _p in self.properties:
            setattr(self,_p,None)

    def set_cell(self,in_cell):
        """
        1. set the three supercell vectors in cartesian coordinates / (A)
        
        Input:
            - in_cell : a list or np.ndarray such that in_cell[ia][ib] is the ibth cartesian 
                        component of the iath cell vector. Expected shape is (N,3) for N atoms.

        Assumed units:
            - Angstrom (A)
        """
        if not self.fast:
            assert (isinstance(in_cell,(list,np.ndarray))),self.errors['E1']
            assert (len(in_cell)==3),self.errors['E1']
            assert (all(len(_v)==3 for _v in in_cell)),self.errors['E1']

        # note, np.array does a deep copy by default
        self.cell = np.array(in_cell,dtype=float,order='C')

    def set_positions(self,in_uvw):
        """
        2. set atom positions in fractional coordinates with respect to the supercell vectors
        
        Input:
            - in_uvw : a list or np.ndarray such that in_uvw[ia][ib] is the fractional component of
                       the ibth cell vector, for the iath atom. Expected shape is (N,3) for N atoms.
        """
        if not self.fast:
            assert (isinstance(in_uvw,(list,np.ndarray))),self.errors['E2']
            assert (all(len(_atom)==3 for _atom in in_uvw)),self.errors['E2']
       
        self.positions = np.array(in_uvw,dtype=float,order='C')

            
    def set_species(self,in_species):
        """
        3. set atom names 
        
        Input:
            - in_species : a list of strings such that in_species[ia] is the name of the iath atom.
                           Expected shape is (N) for N atoms.
        """
        if not self.fast:
            assert isinstance(in_species,list),self.errors['E3']
            assert (all(isinstance(_atom,str) for _atom in in_species)),self.errors['E3']

        self.species = copy.deepcopy(in_species)
    
    def set_energy(self,in_energy):
        """
        4. set the zero Kelvin supercell energy / (eV)

        Input:
            - in_energy : a float 

        Assumed units:
            - Electron volt (eV)
        """
        if not self.fast:
            assert isinstance(in_energy,float),self.errors['E4']

        self.energy = in_energy

    def set_forces(self,in_forces):
        """
        5. set the atom forces / (eV/A) 
        
        Input:
            - in_forces : a list or np.ndarray such that in_forces[ia][ib] is the ibth cartesian
                          component of force, for the iath atom. Expected shape is (N,3) for N atoms.
            
        Assumed units:
            - Electron volt / Angstrom (eV/A)
        """
        if not self.fast:
            assert (isinstance(in_forces,(list,np.ndarray))),self.errors['E5']
            assert (all(len(_frc)==3 for _frc in in_forces)),self.errors['E5']
        
        self.forces = np.array(in_forces,dtype=float,order='C')

    def set_stress(self,in_stress):
        """
        6. set the supercell stress tensor, the NEGATIVE of the pressure tensor / (GPa) 
        
        Input:
            - in_stress : a list or np.ndarray such that in_stress[ia][ib] is 1 of 9 unique stress 
                          components, for the iath and ibth components of cartesian space. Expected
                          shape is (3,3).
        
        Assumed units:
            - 10^9 Pascal (Gpa)
        """
        if not self.fast:
            assert (isinstance(in_stress,(list,np.ndarray))),self.errors['E1']
            assert (all(len(_v)==3 for _v in in_stress)),self.errors['E1']

        self.stress = np.array(in_stress,dtype=float,order='C')

    def set_charge(self,in_charge):
        """
        7. set atom electronic charge / (|e-|)

        Input:
            - in_charge : a list or np.ndarray of floats such that in_charge[ia] is the charge of atom ia. 
                          Positive values correspond to positive charge. Expected shape is (N) for N atoms.

        Assumed units:
            - magnitude of electronic charge (|e-|)
        """
        if not self.fast:
            assert (isinstance(in_charge,(list,np.ndarray))),self.errors['E6']
            assert (all(isinstance(_c,float) for _c in in_charge)),self.errors['E10']

        self.charge = np.array(in_charge,dtype=float)

    def set_name(self,in_name):
        """
        8. set configuration name. Name must be unique as is used as an identifier.

        Input:
            - in_name : a string representing a unique structure name
        """
        if not self.fast:
            assert (isinstance(in_name,str)),self.errors['E3']
        

        self.name = in_name

    def set_bulkmodulus(self,in_bulkmodulus):
        """
        9. set the supercell bulk modulus / (GPa)

        Input:
            - in_bulkmodulus : a float

        Assumed units:
            - 10^9 Pascals (GPa)
        """
        if not self.fast:
            assert(isinstance(in_bulkmodulus,float)),self.errors['E11']

        self.bulkmodulus = in_bulkmodulus

    def set_spacegroup(self,in_spacegroups):
        """
        10. set the spacegroup with 'number' or 'Hermann-Maugin' conventions
        
        Input:
            - in_spacegroups = a dictionary with 'number' and/or 'Hermann-Muagin' as keys,
                               attributes of each key are the crystal space group in each convention
            
        """
        supported_conventions = ['number','Hermann-Mauguin']
        if not self.fast:
            assert (all(_convention in supported_conventions for _convention in in_spacegroups)),self.errors['E12']
            if 'number' in in_spacegroups:
                assert (isinstance(in_spacegroups['number'],int)),self.errors['E6']

        self.spacegroup = copy.deepcopy(in_spacegroups) 
        
    def set_edensity(self,in_edensity):
        """
        11. set the electron density for the given structure

        Input:
            in_edensity:  a list of dictionaries, where each dictionary has 4 attributes: 'x','y','z',
                          'density' where 'x','y','z' are coordinates in a cartesian representation of 
                          the grid point and 'density' is the electron density at 'x','y','z'.
        Units:
            - 'x','y','z' (A)
            - 'density' (unknown - COMPLETE THIS)
        """
        
        if isinstance(in_edensity,list): #assumes that in_edensity are given as a list of dicts
            self.edensity = {"xyz":np.array([[v['x'],v['y'],v['z']] for v in in_edensity],dtype=float),
                             "density":np.array([v["density"] for v in in_edensity])} 
        elif isinstance(in_edensity,dict): #assumes that in_edensity is given as a dict with keys "xyz" and "density" with np.ndarray values
            assert ("xyz" in in_edensity and "density" in in_edensity), "Assertion failed - expected 'xyz' and 'density' as keys, got {} instead!".format(in_edensity.keys())
            assert (len(in_edensity["xyz"])==len(in_edensity["density"])), "Assertion failed - given number of positions doesn't match the number of densities ({}!={})!".format(len(in_edensity["xyz"]),len(in_edensity["density"]))
            self.edensity = {"xyz":np.array(in_edensity["xyz"]), "density":np.array(in_edensity["density"])}    
        else:
            print("Error - expected in_edensity to be either of list or dict, got {} instead!".format(type(in_edensity)))
            raise
    
    def set_files(self,in_files):
        """
        12. set the list of file names contributing to the structure

        Input:
            - in_files = a list of file names contributing to the structure
        """
        if not self.fast:
            # deprecated as want to update structure with multiple files
            #assert self.files is None,"Assertion failed - self.files already set to {} of type {}.".format(self.files,type(self.files))
            assert (isinstance(in_files,list)),assertion_statement(list,"in_files",in_files)
            assert (all(isinstance(_a,str) for _a in in_files)),assertion_statement("list of str","in_files",in_files)
   
        self.files = copy.deepcopy(in_files)
    
    def set_enthalpy(self,in_enthalpy):
        """
        13. set the enthalpy, H = U(0K) + PV

        Input:
            - in_enthalpy, a floating point number 

        Assumed units:
            - electron volt (eV)
        """
        if not self.fast:
            assert (isinstance(in_enthalpy,float)),self.errors['E9']

        self.enthalpy = in_enthalpy

    def get_cell(self):
        """
        1. return supercell vectors / (A)

        Output:
            - get_cell() = a np.ndarray such that get_cell()[ia][ib] is the ibth cartesian component
                           of the iath cell vector. Returned shape is (3,3).

        Returned units:
            - Angstrom (A)
        """
        return self.cell
        
    def get_positions(self, wrap=False):
        """
        2. return fractional atomic positions with respect to the structure cell vectors

        Output:
            - get_positions() : a np.ndarray of np.shape(get_positions()) = [N,3] for an N atom 
                                supercell. get_positions()[i,j] is the fractional coordinate of 
                                the ith atom along the jth cell vector.
        """
        if self.positions is None:
            return None
        
        _pos = np.copy(self.positions)
        if wrap:
            for i in range(3):
                _pos[:,i] = _pos[:,i] % 1.
        
        return _pos

    def get_species(self):
        """
        3. return the species list of atoms in the supercell. assumed ordering is that of 
           atoms in get_positions()[i,j].

        Output: 
            - get_species() : a list of character strings.
        """
        return self.species

    def get_energy(self):
        """
        4. return the approximate 0K energy for the structure / (eV)

        Output:
            - get_energy() : a float

        Returned units:
            - Electron volt (eV)
        """
        return self.energy

    def get_forces(self):
        """
        5. return the atom forces in cartesian coordinates / (A/eV)
        
        Output:
            get_forces() : a np.ndarray of np.shape(get_forces()) = [N,3] for an N atom
                           supercell. get_forces()[i,j] is the jth cartesian component
                           of force on the ith atom. Assumed ordering is that of 
                           get_positions()[i,j].

        Returned units:
            Angstrom / Electron volt (A/eV)
        """
        return self.forces

    def get_stress(self):
        """
        6. return the stress tensor (negative of the pressure tensor) acting
           on the supercell boundary / (GPa)

        Output:
            get_stress() : a np.ndarray of np.shape(get_stress()) = [3,3]. 
                           For physical systems, get_stress()[i,j] = get_stress()[j,i]
                           and i=0,1,2 correspond to the stress component along the 
                           0,1,2nd component of our cartesian reference frame (x,y,z).
                           
        Returned units:
            Giga Pascals (GPa)
        """
        return self.stress

    def get_charge(self):
        """
        7. return the electronic charge of atoms in the structure. assumed ordering is
           that of get_positions()[i,j] / (|e-|)

        Output:
            get_charge() : a list of floating point numbers of length N for a N atom
                           supercell. Negative charge for electrons, positive for protons.

        Returned units:
            |electronic charge| (|e-|)
        """
        return self.charge

    def get_name(self):
        """
        8. return the unique name assigned to this structure. 

        Output:
            get_name() : a string of characters
        """
        return self.name

    def get_bulkmodulus(self):
        """
        9. return the bulk modulus of the structure

        Output:
            get_bulkmodulus() : a float

        Returned units:
            Giga Pascals (GPa)

        """
        return self.bulkmodulus

    def get_spacegroup(self):
        """
        10. return the cyrstal spacegroup

        Output:
            get_spacegroup() : a dictionary {'number':<number> ,'Hermann-Muagin': <Hermann-Muagin>}
                               where <number> and <Hermann-Muagin> is the crystal spacegroup in
                               number and Hermann-Muagin conventions respectively.
        """
        return self.spacegroup

    def get_edensity(self):
        """
        11. return the electron density of the structure

        Output:
            get_edensity() : a list of dictionaries where each dictionary has attributes 
                             'x','y','z','density', giving the electron density at a 
                             specific position (in cartesian coordinates, ) in the supercell
            
        Returned unit:
            - UNKNOWN - NEED TO CHECK THIS BIT FROM CASTEP DOCUMENTATION
        """
        return self.edensity
    def get_files(self):
        """
        12. return the electronic simulation files from which this structure
            informated has been parsed.

        Output:
            get_files() : a list of strings giving the file names of contributing 
                          electronic structure files to this structure.
        """
        return self.files
    def get_enthalpy(self):
        """
        13. return the supercell enthalpy, H = U(0K) + PV /(eV)

        units:
            - electron volt (eV)
        """
        return self.enthalpy

    def get_ase_atoms(self):
        """
        method to return the position, cell and species information as an ase atoms object
        """
        from ase.atoms import Atoms
        from ase.atom import Atom

        # check that all of these properties have been set
        props = ["cell","positions"]

        assert all([getattr(self,_prop) is not None for _prop in props]),\
        '{} attributes must be set before formatting as an ase atoms object'.format(props)

        def _fractocart(gamma,cell):
            return [  sum([gamma[j]*cell[j,i] for j in range(3)]) for i in range(3) ]
            

        atoms_list = []

        for i,_pos in enumerate(self.positions):
            # convert fractional to cartesian coordinates
            _xyztuple = tuple(_fractocart(_pos,self.cell))

            # element
            if self.species:
                _element = self.species[i]
            else:
                _element = None

            # atomic mass
            _mass = None

            # electronic charge
            if self.charge:
                _charge = self.charge[i]
            else:
                _charge = None

            atoms_list.append(Atom(symbol=_element,position=_xyztuple,mass=_mass,charge=_charge))
        return Atoms(atoms_list,cell=self.cell,pbc=True)

    def set_ase_atoms(self,atoms):
        """
        method to parse an ase Atoms object
        """
        from ase.atoms import Atoms

        if not self.fast:
            assert isinstance(atoms,type(Atoms())),self.errors["E13"]

        if atoms.cell is not None:
            self.set_cell(atoms.get_cell())
        if atoms.get_chemical_symbols() is not None:
            self.set_species(atoms.get_chemical_symbols())
        if atoms.positions is not None and atoms.cell is not None:
            # can't set fractional coordinates without cell vector
            self.set_positions(atoms.get_scaled_positions())
        if atoms.get_initial_charges() is not None:
            self.set_charge(atoms.get_initial_charges())

    def __setitem__(self,key,value):
        assert key in self.iproperties, "Assertion failed - got unexpected key {}. Expected one of: {}".format(key,self.iproperties)
        getattr(self,self.set_methods[key])(value)
        
    def __getitem__(self,key):
        assert key in self.iproperties, "Assertion failed - got unexpected key {}. Expected one of: {}".format(key,self.iproperties)
        return getattr(self,self.get_methods[key])()
        
    def __call__(self,**kwargs):
        for key, value in kwargs.items():
            getattr(self,self.set_methods[key])(value)
        
    def keys(self):
        return iter(self.iproperties)
    def values(self):
        return iter([getattr(self,k) for k in self.iproperties])
    def items(self):
        return iter([(k,getattr(self,k)) for k in self.iproperties])
