import sys, os, copy, pickle
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath('../data_structures'))
from parsers import parser_vasp
from parsers import parser_castep
from parsers import parser_profess
from parsers.parser_castep import deprecate
from data_structures import inhouse_formats
from time import time

def write_supercells2pickle(path,supercells):
    # write data from supercell form into a dataframe and write that to disk as a csv file
    print("Writing pickle to {}...".format(path))
    with open(path,'wb') as f:
        pickle.dump(supercells,f)

def read_pickle2supercells(path):
    with open(path,'rb') as f:
        supercells = pickle.load(f)
    
    return supercells

class GeneralInputParser:
    """General parser.
    
    General parser for the end-user. This parser calls DFT code specific parsers
    which have standardized attribute names: "energy", "force", "xyz", "cell" and "species".
    The specific parsers can be called via PARSERNAME(path,file_type='FILETYPE').run().
    
    
    Attributes
    ----------
        
    implemented_file_types : dict
        a dictionary of supported file extensions with the 
        associated dft code

    implemented_dft_codes  : dict
        a dictionary of dft codes and an interface to the
        associated parser. dft code parsers must accept 2 arguments:
        (path,file_type) , where path is a single file given by its
        full path and file_type is the file extension as declared in
        self.implemented_file_types

    parse_file(path) : str
        parse a specific file, 'path', assuming the file extension 
        is recognised in self.implemented_file_types

    parse_all(path) : 
        parse all files in 'path' (a string or list of strings), 
        giving the target directory(ies), the extensions of which 
        are recognised in self.implemented_file_types
    
    Notes
    -----
    This class assumes that each DFT calculation is represented with as
    unique prefix in its name (i.e. structure["name"].split('.')[0] if structure
    is an instance of the class supercell). all files with the same prefix will
    merged.
    This convention may be awkward at first for VASP but should in the end be
    usful for bookkeeping.
    """
    
    def __init__(self,verbose=False):
        self.verbose = verbose
        # supported file types and associated dft code
        self.implemented_file_types = {'xml':'vasp','contcar':'vasp',\
        'md':'castep','geom':'castep','castep':'castep','den_fmt':'castep',\
        'profess':'profess','ion':'profess','final.geom':'profess','den':'profess'}
        
        # interface to dft code parsers
        self.implemented_dft_codes = {'vasp':parser_vasp.parse,'castep':parser_castep.parse,\
        'profess':parser_profess.parse}
        
        #self.implemented_file_types = {'vasp':['vasprun','contcar'],'castep':['md','geom','castep','den_fmt']}
       
        # file extensions must be unique, search for duplicates
        for ia,_typea in enumerate(self.implemented_file_types):
            for ib,_typeb in enumerate(self.implemented_file_types):
                if ia!=ib and _typea==_typeb:
                    raise ValueError('implementation error, duplicate file types: {} {}'.format(_typea,typeb))

        # list of supercell class objects
        self.supercells = None 
        
    def __iter__(self):
        assert self.supercells is not None, "Assertion failed - self.supercells is None!"
        return iter(self.supercells)

    def sort(self):
        #print("before supercells {}".format(self.supercells))
        self.supercells = sorted(self.supercells)
        if self.verbose:
            print("\nfinal supercell names: {}".format(", ".join([v.name for v in self.supercells])))

    def __delitem__(self, key):
        self.__delattr__("supercells")[key]

    def __getitem__(self, key):
        return self.__getattribute__("supercells")[key]
    
    def __len__(self):
        if self.supercells is None:
            return 0
        else:
            return len(self.supercells)
        
    def parse_file(self,path):
        """
        parse a single file, given that the file exists and has a file extension
        consistent with the supported file types.

        Input:

            - path : the full path of a file intended to be parsed

        Output:
            - self.supercells : append path's structure to the current list of
                                structurs in self.supercells
        """

        if os.name != "posix": # win os
            _path = path.replace("\\","/")
        else:
            _path = copy.deepcopy(path)

        assert os.path.exists(_path),'{} does not exist'.format(_path)
        # assume unix system
        #assert (len(path.split('/')[-1].split('.'))==2 and \
        assert \
        ('.'.join(_path.split('/')[-1].split('.')[1:]).lower() in self.implemented_file_types),\
        '{} is not a supported file type: {}'.format(_path,[_a for _a in self.implemented_file_types])
        
        # file type, allow for extensions with one than one '.' delimiter
        file_extension = '.'.join(_path.split('/')[-1].split('.')[1:]).lower()

        """
        pass file's path and extension to appropriate code's parser

        perform internal check that file type is supported and interface to 3rd 
        party parsing codes if appropriate.
        """
        parser_obj = self.implemented_dft_codes[self.implemented_file_types[file_extension]]\
            (_path,file_type=file_extension)

        # parse the file 'path'
        parser_obj.run()

        assert isinstance(parser_obj.supercells,list),'implementation error, parser must return a list'

        # append the parsed structure to self.supercells
        if isinstance(self.supercells,type(None)):
            self.supercells = []
        self.supercells += copy.deepcopy(parser_obj.supercells)
       
        del parser_obj

    def parse_all(self,path,DiscardNoneEnergy=True):
        """
        parse the structure information from all supported file types in 'path', 
        a directory or list of directories

        If DiscardNoneEnergy = True : remove merged structures with no energy value

        Parameters
        ----------    
        path : str or list of str
            a string or list of strings giving all directories from which to 
            parse all supported file types in self.implemented_file_types
        
        Returns
        -------
        self.supercells : 
            append acquired structures to self.supercells, a list of all parsed structures
                                
        Notes
        -----
        Important: When setting the name of the supercell make sure its prefix 
        is unique when reading multiple files such that each complete structure
        can be constructed from all supercells with identical prefix and varying suffix.
        """


        if isinstance(path,str):
            pass
            # force "/" ending to path
            # if path[-1]!='/':
            #     path += '/'
            
            directory_list = [path]
        elif isinstance(path,list):
            assert all(isinstance(_a,str) for _a in path),'Assertion failed - expected a list of strings, got {} instead.'.format(path)
            
            # for i,_dir in enumerate(path):
            #     # force "/" ending to path
            #     if _dir[-1]!='/':
            #         path[i] += '/'
            
            directory_list = copy.deepcopy(path)
        else:
            raise ValueError("Paremeter 'path' ({}) must be a string or list of strings".format(path))
            
        # loop over all target directories
        t0 = time()
        for ia,_dir in enumerate(directory_list):
            # ensure directory path finishes with "/" - assume unix
            # if _dir[-1]!='/':
            #     _dir += '/'

            # check _dir exists!
            assert os.path.isdir(_dir),"Assertion failed - specified path ({}) is not a directory!".format(_dir)

            # list of files in _dir
            tmpfiles = [_f for _f in os.listdir(_dir) if os.path.isfile(os.path.join(_dir,_f))]
    
            # include only files with a delimator
            files = [_f for _f in tmpfiles if len(_f.split('.'))>=2]

            # check for anomylous delimators
            #assert all(len(_f.split('.'))==2 for _f in files),'all files must have a single "." delimator'

            files = sorted(files)
            for _f in files:
                # allow for double '.' delimited file suffixes!
                file_extension = '.'.join(_f.split('.')[1:])
                
                if file_extension in self.implemented_file_types:
                    if self.verbose:
                        print("\nprocessing: {}".format(_f))
                    # create instance of dft code parser for _f
                    parser_obj = self.implemented_dft_codes[self.implemented_file_types[file_extension]]\
                        (directory_list[ia]+_f,file_type=file_extension)

                    # parse _f
                    parser_obj.run(verbose=self.verbose)
        
                    #assert isinstance(parser_obj.supercells,list),\
                    #    'implementation error, parser must return a list'

                    # append to self.supercells
                    if self.supercells is None:
                        self.supercells = []
                    self.supercells.extend(copy.deepcopy(parser_obj.supercells))
                    
        if self.verbose:
            print("looping all directories {} s...".format(time()-t0))
        # search for and merge segments of the same structure, overwriting self.supercells
        t0 = time()
        #self.merge_supercells(DiscardNoneEnergy=DiscardNoneEnergy)
        self.merge_supercells()
        if self.verbose:
            print("merging supercells {} s...".format(time()-t0))

    def get_supercells(self):
        return self.supercells

    def merge_supercells(self):
        """
        list of structures with same name into a new structure
        """
        def _sort_file_order(idxs,code):
            """
            sort file idx order based upon order_of_precedence 
            """
            
            # dictionary with idx as key, min. file precedence as value
            tmp = {}

            for _idx in idxs:
                vals = [order_of_precedence[code]['.'.join(_file.split('.')[1:])] for _file in self.supercells[_idx]["files"]]
        
                tmp.update({_idx:min(vals)})
           
            # return ordered list of idxs for file reading
            return [_a[0] for _a in sorted(tmp.items(),key = lambda x:x[1]) ]


        # file suffixes with lower index have higher precedence when merging
        order_of_precedence = {'castep':{"castep":2,"den_fmt":3,"md":1,"geom":0},
                               'profess':{"profess":0,"den":1,"final.geom":2,"ion":3}}
       
        # attributes requiring special consideration when merging
        attribute_exceptions = ['name','files']

        # set of all seednames in self.supercells
        seednames = set([_structure["name"] for _structure in self.supercells])

        # check for trivial cases
        if len(seednames)==len(self.supercells):
            # structures are already fully merged
            return

        # list for new merged supercells
        new_supercells = [None]*len(seednames)
        
        for i,_seedname in enumerate(seednames):
            # lambda to help pick structures with same name as _seedname
            g = lambda x,y,z:x if y==z else -1

            # idx of structures in self.supercells with same seedname != -1
            connected_structures = [g(_struct[0],_struct[1]["name"],_seedname) for _struct in enumerate(self.supercells)]
            connected_structures = set(connected_structures)

            # leave idx's of only structures connected by name to _seedname
            if -1 in connected_structures: # without this reading single castep calculations crashed...
                connected_structures.remove(-1)

            # list for codes contributing to files associated with _seedname
            contributing_codes = []

            # check which code contributing files are from
            for _idx in connected_structures:
                # list of file suffixes for _struct
                suffixes = ['.'.join(_file.split('.')[1:]) for _file in self.supercells[_idx]["files"]]

                contributing_codes += [self.implemented_file_types[_s] for _s in suffixes]

            assert len(set(contributing_codes))==1,'files can only be merged if from the same code.'

            # code from which files for _seedname originate
            contributing_code = contributing_codes[0]
            
            # sort idx's according to file precedences
            connected_structures = _sort_file_order(connected_structures,contributing_code)

            # instantiate new supercell
            new_supercells[i] = inhouse_formats.supercell()
                
            for _attribute in new_supercells[i].properties:
                # loop over subset of (unset) attributes
                if _attribute not in attribute_exceptions and new_supercells[i][_attribute] is None:
                    
                    # loop over contributing structures in order of file precedence
                    for _idx in connected_structures:
                        if new_supercells[i][_attribute] is None and \
                        self.supercells[_idx][_attribute] is not None:
                            new_supercells[i][_attribute] = copy.deepcopy(self.supercells[_idx][_attribute])
                            
            # structure name
            new_supercells[i]["name"] = copy.deepcopy(self.supercells[connected_structures[0]]["name"])

            tmp_files = []

            for _idx in connected_structures:
                tmp_files += self.supercells[_idx]["files"]

            # contributing files
            new_supercells[i]["files"] = list(set(tmp_files))
        

        # overwrite previous supercells
        self.supercells = None
        self.supercells = new_supercells



    def merge_supercells_deprecated(self,new=True,DiscardNoneEnergy=True): #ONLY SUPPORTS CASTEP!
        """
        determine if any structures in self.supercells are duplicates, or if a number
        of structures contains segments of information from the same structure.

        merge any segmented or duplicte structures into a new distint supercell object, 
        overwriting the self.supercells from file reading.
        
        Components:
            - merge_group(group,structure_name) : merges a list of supercell objects 'group'
                                                  into a single object of name 'structure_name'.
            
            - sort_groups(structures)           : sort a list of structure objects, 'structures'
                                                  into a list of lists of dictionaries containing
                                                  the element index and object name in 'structures'
                                                  that belong to a given group.
                - unequal(float1,float2)        : return true if float1 != float2, false otherwise
        """
        
        assert isinstance(getattr(self,'supercells'),list),"Assertion failed - expected the attribute to be a list, got {} instead!".format(type(self.supercells))
        
        # sort structure objects into distinct structures
        def _merge_castep(name,idx_dict,keys): 
            """
            merge by matching structure.name prefixes assigned by parser
            """
            
            s = inhouse_formats.supercell(fast=False)
            s["name"] = name
            s["files"] = []
           
            # .md and .geom files have a much high precision for forces and position than castep file!
            file_order = {"castep":2,"den_fmt":3,"md":1,"geom":0}
            
            idx_sorted = sorted(idx_dict[name], key = lambda x: file_order[self.supercells[x]["name"].split('.')[-1]])
            for ix in idx_sorted:
                for key in keys:
                    value = self.supercells[ix][key]
                    if value is not None:
                        if key == "files":
                            s["files"] = [_a for _a in set(s["files"]+value)]
                            #s["files"].append(value[:])
                        if key == "energy":
                            if self.supercells[ix]["name"].split('.')[-1]=="castep":
                                try: #writing only those values to supercell which are not yet known
                                    s[key] = value
                                except:
                                    pass 
                        elif key!="files":
                            try:
                                s[key] = value
                            except:
                                pass
            return s
        def _merge_profess(name,idx_dict,keys): 
            """
            merge by matching structure.name prefixes assigned by parser
            
            For prefix, use only the FIRST '.' delimited string of a name attribute.

            For suffix, use REMAINDER of '.' delimited string of name attribute
            
            eg:
                _name = 'calc-1-3.final.geom'
                    -> prefix = calc-1-3
                    -> suffix = final.geom
            """
            
            s = inhouse_formats.supercell(fast=False)
            s["name"] = name
            s["files"] = []
            file_order = {"profess":0,"den":1,"final.geom":2,"ion":3}
            
            idx_sorted = sorted(idx_dict[name], key = lambda x: file_order[\
            '.'.join(self.supercells[x]["name"].split('.')[1:])])

            for ix in idx_sorted:
                for key in keys:
                    value = self.supercells[ix][key]
                    if value is not None:
                        if key == "files":
                            s["files"] = [_a for _a in set(s["files"]+value)]
                            #s["files"].append(value[:])
                        if key == "energy":
                            if self.supercells[ix]["name"].split('.')[-1]=="profess":
                                try: #writing only those values to supercell which are not yet known
                                    s[key] = value
                                except:
                                    pass 
                        elif key!="files":
                            try:
                                s[key] = value
                            except:
                                pass
            return s
        
        merge_funs = {"castep":_merge_castep,"profess":_merge_profess}
        
        if new:
            num_scells = len(self.supercells)
            # use first '.' delimited string ONLY for prefix
            unique_prefixes = [s["name"].split('.')[0] for s in self.supercells]
            idx_dict = dict()
            for i in range(num_scells):
                name = unique_prefixes[i]
                
                #group supercell objects by name prefix 
                if name in idx_dict:
                    idx_dict[name].append(i)
                else:
                    idx_dict[name] = [i]
            unique_prefixes = set(unique_prefixes)
            unique_structures = {name:None for name in unique_prefixes}
            keys = set(self.supercells[0].keys()).difference(["name"])#,"files"])
            unique_prefixes = sorted(list(unique_prefixes))
            

            for i,name in enumerate(unique_prefixes):
                #this is a bit indirect but does: get the suffix of the first listed file associated with the unique prefix and fetches the dft code
                #this is then used to merge the data from the associated files appropriately 
                dft_code_suffix = '.'.join(self.supercells[idx_dict[name][0]]["name"].split('.')[1:])
                unique_structures[name] = merge_funs[self.implemented_file_types[dft_code_suffix]](name,idx_dict,keys)
                
            if DiscardNoneEnergy:
                #store the merged supercells removing all without energy values
                self.supercells = [unique_structures[val] for val in unique_prefixes if unique_structures[val]["energy"] is not None]
            else:
                #keep all merged structures
                self.supercells = [unique_structures[val] for val in unique_prefixes]

        else:
            import gc

            groups = sort_groups(self.supercells)

            # remove duplicate groups
            glist = []

            for ia,_g in enumerate(groups):
                # produce list of lists of structure indices
                glist += [[[___g for ___g in __g][0] for __g in _g]]
                glist[ia].sort()

            groups = []

            # form list of distinct lists of structures indices
            for ia,_group in enumerate(glist):
                if _group not in groups:
                    groups += [_group]

            # form list of groups (a list) of unique structures 
            unique_structures = []

            # query all structure names on the stack and carry on from laste "structure_x" x value
            stack_names = []
            for obj in gc.get_objects():
                if (isinstance(obj,inhouse_formats.supercell)):
                    # check for supercells created by merge_supercells()
                    if getattr(obj,obj.get_methods['name'])().split('_')[0]=='structure':
                        stack_names.append(getattr(obj,obj.get_methods['name'])())
            # if first time merge_supercells() has been called
            if len(stack_names)==0:
                xval = 0
            else:
                xval = max([int(_name.split('_')[1]) for _name in stack_names])

            for ia in range(len(groups)):
                tmp = []
                for _g in groups[ia]:
                    tmp += [self.supercells[_g]]
                unique_structures += [merge_group(tmp,'structure_'+str(xval+ia+1))]
                del tmp
            # overwrite old file structures from stack
            self.supercells = copy.deepcopy(unique_structures)

            # free unused memory
            del unique_structures
        
