"""
dummy module such that Python will cleanly run if fortran has not been compiled
"""

message = 'Fortran has not been compiled. Please read install guide or use pure Python for now'

class f90Error(Exception):
    def __init__(self,message):
        self.message = message
    def __repr__(self):
        return repr(self.message)

def f90wrap_get_num_threads():  
    raise f90Error('get_num_threads : '+message)

def f90wrap_query_ball_point(*args):
    raise f90Error('query_ball_point : '+message)

def f90wrap_meam_bond_generator(*args):
    raise f90Error('MEAM_aniso_bonds : '+message)

def f90wrap_wrapped_cos(*args):
    raise f90Error('wrapped_cos : '+message)

def f90wrap_wrapped_smooth(*args):
    raise f90Error('wrapped_smooth : '+message)

def f90wrap_openmp_isotropic_phi_cos(*args):    
    raise f90Error('isotropic_phi : '+message)

def f90wrap_isotropic_phi_cos(*args):    
    raise f90Error('isotropic_phi : '+message)

def f90wrap_openmp_anisotropic_phi_cos(*args):    
    raise f90Error('aniisotropic_phi : '+message)

def f90wrap_anisotropic_phi_cos(*args):    
    raise f90Error('aniisotropic_phi : '+message)
    
