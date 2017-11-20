from scoop import shared
import numpy as np

def var(k,key=None,key2=None):
    if key is None:
        return shared.getConst(k)
    else:
        if key2 is None:
            return shared.getConst(k)[key]
        else:
            return shared.getConst(k)[key][key2]

def setup_emb_and_pair_contribution():
    params, smooth_emb = var("data",key="params"), var("data",key="smooth_emb")
    
    if smooth_emb:
        emb_n = [np.array([[params["fprho_n"][s][n][j] * (params["psiprho"][s][n]*params["coskrho"][s][:,n] - params["psirho"][s][n] * params["ksinkrho"][s][:,n]) \
                for j in range(3)] for n in range(params["N_atoms"][s])])\
                for s in range(params["N_bonds"])] #(Natoms,Nk,dim)
        emb_i = [[np.array([[params["fprho_i"][s][n][i][j] * (params["psiprho"][s][params["neigh_idx_super"][s][n][i]]*params["coskrho"][s][:,params["neigh_idx_super"][s][n][i]] - params["psirho"][s][params["neigh_idx_super"][s][n][i]]*params["ksinkrho"][s][:,params["neigh_idx_super"][s][n][i]])\
                for j in range(3)] for i in range(params["fprho_i"][s][n].shape[0])]) \
                for n in range(params["N_atoms"][s])] \
                for s in range(params["N_bonds"])]
    else:
        emb_n = [np.array([[params["fprho_n"][s][n][j] * ( - params["ksinkrho"][s][:,n]) \
                    for j in range(3)] for n in range(params["N_atoms"][s])])\
                    for s in range(params["N_bonds"])] #(Natoms,Nk,dim)
        emb_i = [[np.array([[params["fprho_i"][s][n][i][j] * (- params["ksinkrho"][s][:,params["neigh_idx_super"][s][n][i]])\
                    for j in range(3)] for i in range(params["fprho_i"][s][n].shape[0])]) \
                    for n in range(params["N_atoms"][s])] \
                    for s in range(params["N_bonds"])]
    pair_n = [[np.array([[params["r_vec"][s][n][i][j] * (params["psipr"][s][n][i]*params["coskr"][s][n][:,i] - params["psir"][s][n][i]*params["ksinkr"][s][n][:,i])\
                    for j in range(3)] for i in range(params["r_vec"][s][n].shape[0])]) \
                    for n in range(params["N_atoms"][s])] \
                    for s in range(params["N_bonds"])]
    return emb_n, emb_i, pair_n

def get_mappers(map_key="mapper"):
    """Generates lists of lists of np.ndarrays of int which point to specific places in the weights
    array related to the pair/embedding type contribution and the respective element of structure s and 
    atom n.

    """
    params,mapper = var("data",key="params"), var("data",key=map_key)
    
    pair_map = [[[] for n in range(params["N_atoms"][s])] for s in range(params["N_bonds"])]
    emb_neigh_map = [[[] for n in range(params["N_atoms"][s])] for s in range(params["N_bonds"])]
    emb_map = [[] for s in range(params["N_bonds"])]
    for s in range(params["N_bonds"]):
        emb_map[s] = np.array([mapper["emb"][params["species"][s][v_n]] for v_n in range(params["N_atoms"][s])])
        for n in range(params["N_atoms"][s]):
            pair_map[s][n] = np.array([mapper["pair"][params["pair_species"][s][n][v_i]] \
                            for v_i in range(params["coskr"][s][n].shape[1])])
            emb_neigh_map[s][n] = np.array([mapper["emb"][params["emb_species"][s][n][i]] \
                                for i in  range(params["fprho_i"][s][n].shape[0])])
    return pair_map, emb_neigh_map, emb_map

def get_pair_mappers():
    """Generates lists of lists of np.ndarrays of int which point to specific places in the weights
    array related to the PAIR type contribution ONLY and the respective element of structure s and 
    atom n.

    """
    params,mapper = var("data",key="params"), var("data",key="mapper")
    
    pair_map = dict()
    for s in range(params["N_bonds"]):
        for n in range(params["N_atoms"][s]):
            for j in range(params["coskr"][s][n].shape[1]):
                el0, el1 = params["pair_species"][s][n][j]
                if el0 != el1:
                    pair_map[(s,n,j)] = mapper["pair"][params["pair_species"][s][n][j]]

    return pair_map

def get_embmod_mappers():
    """Generates lists of lists of np.ndarrays of int which point to specific places in the weights
    array related to the PAIR type contribution and embedding densities (for beyond single element systems) for 
    the respective element of structure s and atom n.

    """
    params, mapper = var("data",key="params"), var("data",key="mapper")
    print("\nparams[species] ",params["species"])
    print("\nparams[N_atoms] ",params["N_atoms"])
    print("\nmapper[emb] ",mapper["emb"])
    
    pair_map = dict()
    for s in range(params["N_bonds"]):
        for n in range(params["N_atoms"][s]):
            for j in range(params["coskr"][s][n].shape[1]):
                el0, el1 = params["pair_species"][s][n][j]
                if el0 != el1:
                    pair_map[(s,n,j)] = mapper["pair"][params["pair_species"][s][n][j]]
                    
    emb_neigh_map = [[[] for n in range(params["N_atoms"][s])] for s in range(params["N_bonds"])]
    emb_map = [[] for s in range(params["N_bonds"])]
    for s in range(params["N_bonds"]):
        emb_map[s] = np.array([mapper["emb"][params["species"][s][v_n]] for v_n in range(params["N_atoms"][s])])
        for n in range(params["N_atoms"][s]):
            emb_neigh_map[s][n] = np.array([mapper["emb"][params["emb_species"][s][n][i]] \
                                for i in  range(params["fprho_i"][s][n].shape[0])])

    return pair_map, emb_neigh_map, emb_map

def setup_weights(tmp_x,key_emb="emb_map",key_pair="pair_map",key_embn="emb_neigh_map",embmod=False):
	"""Sets up the weights in form of arrays using the previously created mapper with get_mapper().

	"""
	if embmod:
		pair_map = var(key_pair)
		N_atoms = var("data",key="params",key2="N_atoms")
		N_bonds = var("data",key="params",key2="N_bonds")
		_W_pair = [[np.take(tmp_x,pair_map[s][n],axis=0)
			    for n in range(N_atoms[s])] \
			    for s in range(N_bonds)]
			    		
		return _W_pair
	else:
		emb_map,pair_map,emb_neigh_map = var(key_emb), var(key_pair), var(key_embn)
		N_atoms = var("data",key="params",key2="N_atoms")
		N_bonds = var("data",key="params",key2="N_bonds")
		_W_emb = [np.take(tmp_x,emb_map[s],axis=0) for s in range(N_bonds)]
		_W_pair = [[np.take(tmp_x,pair_map[s][n],axis=0)
			    for n in range(N_atoms[s])] \
			    for s in range(N_bonds)]
		_W_emb_neigh = [[np.take(tmp_x,emb_neigh_map[s][n],axis=0)
			    for n in range(N_atoms[s])] \
			    for s in range(N_bonds)]
	
		return _W_emb,_W_pair,_W_emb_neigh

def setup_weights_pair(tmp_x,key_pair="pair_map"): # sets up the weights for pair contributions only
    pair_map = var(key_pair)
    N_atoms = var("data",key="params",key2="N_atoms")
    N_bonds = var("data",key="params",key2="N_bonds")
    _W_pair = var("W_pair_c")
    
    for (s,n,j) in pair_map:
        _W_pair[s][n][j,:] = np.take(tmp_x,pair_map[(s,n,j)],axis=0)
    
    return _W_pair
    
def setup_weights_pair_embmod(tmp_x,key_pair="pair_map",key_emb="emb_map",key_embn="emb_neigh_map"): # sets up the weights for beyond single element fits allowing for fitting of the embedding contribution
	pair_map = var(key_pair)
	N_atoms = var("data",key="params",key2="N_atoms")
	N_bonds = var("data",key="params",key2="N_bonds")
	_W_pair = var("W_pair_c")

	for (s,n,j) in pair_map:
		_W_pair[s][n][j,:] = np.take(tmp_x,pair_map[(s,n,j)],axis=0)
		
	emb_map,emb_neigh_map = var(key_emb), var(key_embn)
	_W_emb = [np.take(tmp_x,emb_map[s],axis=0) for s in range(N_bonds)]
	_W_emb_neigh = [[np.take(tmp_x,emb_neigh_map[s][n],axis=0)
		    for n in range(N_atoms[s])] \
		    for s in range(N_bonds)]

	return _W_emb,_W_pair,_W_emb_neigh