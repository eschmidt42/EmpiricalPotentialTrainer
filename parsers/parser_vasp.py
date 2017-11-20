#import sys, os
#sys.path.insert(0, os.path.abspath('..'))
import os, sys, copy, time, warnings, pickle
import numpy as np
from collections import OrderedDict
from ase.io.vasp import read_vasp
from ase import Atoms
from ase.lattice.spacegroup import crystal
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath('../data_structures'))

# from data_structures import atom_config
from data_structures.inhouse_formats import supercell

def GetVASPRUNdata(path,index=-1,verbose=False):
    from ase.calculators.singlepoint import SinglePointCalculator 
    tree = ET.iterparse(path)
    atoms_init = None 
    calculation = []
    try:
        for event, elem in list(tree): 
            if elem.tag == 'atominfo': 
                species = [] 
                
                for entry in elem.find("array[@name='atoms']/set"): 
                    species.append(entry[0].text.strip()) 
                natoms = len(species) 
                #print 'species ',natoms
                
            elif (elem.tag == 'structure' and elem.attrib.get('name') == 'initialpos'): 
                cell_init = np.zeros((3, 3), dtype=float) 

                for i, v in enumerate(elem.find("crystal/varray[@name='basis']")): 
                    cell_init[i] = np.array([float(val) for val in v.text.split()]) 

                scpos_init = np.zeros((natoms, 3), dtype=float) 
                for i, v in enumerate(elem.find("varray[@name='positions']")): 
                    scpos_init[i] = np.array([float(val) for val in v.text.split()]) 

                constraints = [] 
                fixed_indices = [] 

                for i, entry in enumerate(elem.findall("varray[@name='selective']/v")): 
                    flags = (np.array(entry.text.split() == np.array(['F', 'F', 'F']))) 
                    if flags.all(): 
                        fixed_indices.append(i) 
                    elif flags.any(): 
                        constraints.append(FixScaled(cell, i, flags)) 

                if fixed_indices: 
                    constraints.append(FixAtoms(fixed_indices)) 

                atoms_init = Atoms(species, 
                                 cell=cell_init, 
                                 scaled_positions=scpos_init, 
                                 constraint=constraints, 
                                 pbc=True) 
            elif elem.tag == 'calculation': 
                calculation.append(elem) 
    except ET.ParseError as parse_error: 
        if atoms_init is None: 
            raise parse_error 
        elif not calculation: 
            yield atoms_init

    if calculation: 
        if verbose:
            print('{} MD DFT steps found...'.format(len(calculation)))
        if isinstance(index, int): 
            print('instance')
            steps = [calculation[index]] 
        elif type(index)==list and all(map(lambda v: type(v)==int,index)):
            print('list')
            steps = [calculation[v] for v in index]
        elif index=='all':
            steps = calculation
        else: 
            print('not instance')
            steps = calculation[index] 
    else: 
        steps = [] 

    storage_energy = []
    storage_forces = []
    storage_cells = []
    storage_positions = []
    storage_elements = []
    
    for step in list(steps): 
        
        lastscf = step.findall('scstep/energy')[-1] 

        de = (float(lastscf.find('i[@name="e_0_energy"]').text) 
              - float(lastscf.find('i[@name="e_fr_energy"]').text)) 

        energy = float(step.find('energy/i[@name="e_fr_energy"]').text) + de 
        #print 'energy ',energy

        cell = np.zeros((3, 3), dtype=float) 
        for i, vector in enumerate(step.find('structure/crystal/varray[@name="basis"]')): 
            cell[i] = np.array([float(val) for val in vector.text.split()]) 
        
        #print 'cell ',cell
        storage_cells += [cell]
        
        scpos = np.zeros((natoms, 3), dtype=float) 
        for i, vector in enumerate(step.find('structure/varray[@name="positions"]')): 
            scpos[i] = np.array([float(val) for val in vector.text.split()]) 

        forces = None 
        fblocks = step.find('varray[@name="forces"]') 
        if fblocks is not None: 
            forces = np.zeros((natoms, 3), dtype=float) 
            for i, vector in enumerate(fblocks): 
                forces[i] = np.array([float(val) for val in vector.text.split()]) 
                
        storage_energy += [energy]
        storage_forces += [forces]
        storage_positions += [scpos]
        #print 'storage_cells ',storage_cells
        
        atoms = atoms_init.copy() 
        atoms.set_cell(cell) 
        atoms.set_scaled_positions(scpos) 
        atoms.set_calculator( 
            SinglePointCalculator(atoms, energy=energy, forces=forces))
        yield atoms    

def VASP_parser(dft_path):
    data = GetVASPRUNdata(dft_path,index='all')
    atoms = [v for v in data]
    energies = [v.get_total_energy() for v in atoms]
    forces = [v.get_forces() for v in atoms]
    positions = [v.get_positions() for v in atoms]
    cells = [v.get_cell() for v in atoms]
    elements = [v.get_chemical_symbols() for v in atoms]
    return positions, elements, cells, forces, energies, atoms
    
class parse:
    def __init__(self,path,file_type='contcar'):
        from data_structures.inhouse_formats import supercell
        #self.implemented_filetypes = ['contcar','vasprun']
        self.implemented_filetypes = ['contcar','xml']

        #loading related
        assert (file_type in self.implemented_filetypes), "Assertion failed - got unexpected file_type '{}'! Expected one out of: {}".format(file_type,self.implemented_filetypes)
        self.file_type = file_type
        self.path = path
        
        #data structure related - attributes will be set using the keys of 'self.data_properties'
        self.ase_get_methods = {"energy":"get_total_energy",
                            "forces":"get_forces",
                            "positions":"get_positions",
                            "cell":"get_cell",
                            "species":"get_chemical_symbols",
                            "stress":"get_stress",
                            "charge":"get_charges",
                            "name":"get_name",
                            "bulkmodulus":"get_bulkmodulus",
                            "spacegroup":"get_spacegroup"}
        
        self.supercells = None
            
    def clean(self):
        self.supercells = []
    
    def run(self,verbose=False):
        self.supercells = []
        if self.file_type == 'contcar':
            crys_list = [read_vasp(filename=self.path)]
        #elif self.file_type == 'vasprun':
        elif self.file_type == 'xml':
            crys_list = [v for v in GetVASPRUNdata(self.path,verbose=verbose)]
            
        for ia,crys in enumerate(crys_list):
            a = supercell()
            for k in self.ase_get_methods.keys():
                
                try:                   
                    vals = getattr(crys,self.ase_get_methods[k])()
                    get_success = True
                except:
                    warnings.warn("Warning - unsuccessfully tried to get '{}' from the ase object. Skipping...".format(k))
                    get_success = False
                if get_success:
                    getattr(a,a.set_methods[k])(vals)
                # set name if hasnt been set already
                if isinstance(getattr(a,a.get_methods['name'])(),type(None)):
                    # assume unix-based system
                    getattr(a,a.set_methods['name'])(self.path.split('/')[-1]+'_'+str(ia+1))
                # set file name if it hasnt been done already
                if isinstance(getattr(a,a.get_methods['files'])(),type(None)):
                    getattr(a,a.set_methods['files'])([self.path.split('/')[-1]])
            self.supercells += [a]
      
    def get_supercells(self):
        """
        Returns:
            supercells - list of 'supercell' instances
        """
        return self.supercells
