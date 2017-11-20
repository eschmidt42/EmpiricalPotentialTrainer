from data_structures.inhouse_formats import supercell
import copy
import numpy as np 
import datetime

class wrap_inhouse:
    """
    wrap inhouse supercell class to provide output to DLPOLY v2. CONFIG 
    """

    def __init__(self,structure,seedname=None):
        assert isinstance(structure,supercell),'structure must be an in-house supercell object'

        self.supercell = copy.deepcopy(structure)

        if seedname is not None:
            self.supercell["name"] = seedname

    def write_config(self,seedname=None):
        """
        method to write structure to a DLPOLY version 2.x CONFIG file
        
        NOTE
        ----

        DLPOLY 2.x assumes that fractional coordinates are between [-0.5,0.5]. 
        Shift all fractional coordinates by -0.5 before outputting to cartesians
        """
        
        def _checks(self,seedname):
            assert isinstance(seedname,str),'seedname must be a string'
            assert len(self.supercell["species"])==len(self.supercell["positions"]),'structure incomplete'
            assert cell is not None,'cell vectors must be specified'
        if seedname is not None:
            prefix = seedname
        else:
            if self.supercell["name"] is not None:
                prefix = self.supercell["name"]
            else:
                prefix = 'unnamed'

        # file header
        flines = ['{} automatically generated CONFIG file {}\n'.format(datetime.datetime.today(),prefix),\
                '{:<10}{:<10}{:<10}\n'.format(0,2,len(self.supercell["species"]))]
        
        # cell vectors
        for i in range(3):
            flines.append('{:<20f}{:<20f}{:<20f}\n'.format(self.supercell["cell"][i][0],\
                    self.supercell["cell"][i][1],self.supercell["cell"][i][2]))

        # atom type and position
        for i in range(len(self.supercell["species"])):
            # fractional to cartesian coordinates
            cart = np.dot(self.supercell["positions"][i]-0.5*np.ones(3),self.supercell["cell"])
            
            flines.append('{:<8}{:<10}\n'.format(self.supercell["species"][i],i+1))
            flines.append('{:<20f}{:<20f}{:<20f}\n'.format(cart[0],cart[1],cart[2]))


        with open(prefix+'.config','w') as f:
            f.writelines(flines)
