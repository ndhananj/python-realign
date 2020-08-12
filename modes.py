
from gmx_file_processing import *
from alignment import *


import matplotlib.pyplot as plt

stat_items=['x_coord', 'y_coord', 'z_coord']

def save_matrix(filename,m):
    with open(filename,'wb') as f:
        np.save(f,m)

def load_matrix(filename):
    m = None
    with open(filename,'rb') as f:
        m=np.load(f,m)
    return m

def get_xvg_stats(xvgfile,fitfile=None,unbias=False):
    xvg=read_xvg(xvgfile)
    df = xvg['data']
    coords = df.filter(items=xvg['yaxis labels']).to_numpy()*10 #nm to A
    if(fitfile):
        print("Fitting...")
        src_cs_shape = (coords.shape[0],int(coords.shape[1]/3),3)
        src_cs = coords.reshape(src_cs_shape)
        pdb = PandasPdb()
        pdb.read_pdb(fitfile)
        trg_c = pdb.df['ATOM'].filter(items=stat_items).to_numpy()
        for i in range(src_cs_shape[0]):
            src_mu, trg_mu, rot_mat = find_coords_align(src_cs[i],trg_c,\
                unbias=False,force_mirror=False,force_no_mirror=False)
            src_cs[i] = realign_coords(src_cs[i],src_mu, trg_mu, rot_mat)
        coords = src_cs.reshape((coords.shape[0],coords.shape[1]))
    mean1, mean2, cov, s, u, v = calc_coord_stats(coords,coords,unbias=unbias)
    return mean1, mean2, cov, s, u, v

def shift_by_mode(df,mode,indeces,mul):
    stat_items=['x_coord', 'y_coord', 'z_coord']
    to_change = df.iloc[match_col_in_int_list(df,'atom_number',indeces)]
    coords = to_change[stat_items]
    coords += mode*float(mul)
    to_ret = to_change.copy()
    to_ret[stat_items] = coords
    return to_ret

# eignevector should be in nx3 form
def get_atom_participation_from_eigenvector(S):
    return np.sum(S**2,axis=1)

# normalize and get scaled values
def get_coloring(P):
    m = np.max(P)
    return np.cbrt(P/m)

# Will retunr in KJ/mol/Angtrom^2 assuming D is in Angtrom^2
def spring_constants_from_variances(D,T=293.1):
    R=8.3145
    return R*T/D

def modes(xvgfile,ndxfile,pdbfile,mode_idx,newpdbfile,mul,\
    fit_using_pdb=True,ndx_name='C-alpha',movie_steps=400):
    fitfile = pdbfile if fit_using_pdb else None
    mean1, mean2, cov, s, u, v = get_xvg_stats(xvgfile,fitfile=fitfile)
    shift_shape = (int(u.shape[1]/3),3)
    mode = u[:,int(mode_idx)].reshape(shift_shape)
    print("u[:,",mode_idx,"] =",mode)
    ndx=read_ndx(ndxfile)
    print(ndx[ndx_name])
    ppdb=PandasPdb()
    ppdb.read_pdb(pdbfile)
    ts=np.linspace(-float(0),float(movie_steps),num=movie_steps+1)
    muls=float(mul)*np.cos(2*np.pi*ts/movie_steps)
    for i in range(len(muls)):
        new_df = shift_by_mode(ppdb.df['ATOM'],mode,ndx[ndx_name],muls[i])
        mode_pdb = PandasPdb()
        mode_pdb.df['ATOM'] = new_df
        mode_pdb.to_pdb(path=newpdbfile+str(i)+'.pdb',\
            records=['ATOM'], gz=False, append_newline=True)
