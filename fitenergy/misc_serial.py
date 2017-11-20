import numpy as np

def setup_emb_and_pair_contribution(data):
    
    params = data["params"]
    smooth_emb = data["smooth_emb"]
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

def get_mappers(data):
    params,mapper = data["params"], data["mapper"]

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

def setup_weights(tmp_x,emb_map,pair_map,emb_neigh_map,params):
    
    _W_emb = [np.take(tmp_x,emb_map[s],axis=0) for s in range(params["N_bonds"])]
    _W_pair = [[np.take(tmp_x,pair_map[s][n],axis=0)
            for n in range(params["N_atoms"][s])] \
            for s in range(params["N_bonds"])]
    _W_emb_neigh = [[np.take(tmp_x,emb_neigh_map[s][n],axis=0)
            for n in range(params["N_atoms"][s])] \
            for s in range(params["N_bonds"])]
    
    return _W_emb,_W_pair,_W_emb_neigh