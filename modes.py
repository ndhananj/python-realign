
from gmx_file_processing import *
from alignment import *


import matplotlib.pyplot as plt

stat_items=['x_coord', 'y_coord', 'z_coord']

color_items=['b_factor']

residue_masses={
"ALA": 55.08,
"ARG": 140.19,
"ASN": 98.105,
"ASP": 99.089,
"GLN": 112.132,
"GLU": 113.116,
"GLY": 41.053,
"HIS": 121.143,
"ILE": 97.161,
"LEU": 97.161,
"LYS": 112.176,
"MET": 115.194,
"PHE": 131.178,
"PRO": 81.118,
"SER": 71.079,
"THR": 85.106,
"TRP": 170.215,
"TYR": 147.177,
"VAL": 83.134
}

def get_masses_from_pdb_by_resn(pdb):
    residues=pdb.df['ATOM'][['residue_name']].to_numpy().tolist()
    masses=[residue_masses[residue[0]] for residue in residues]
    return np.array(masses)

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

# eignevector should be in nx3 form for single eigenvector
def get_atom_participation_from_eigenvector(S):
    return np.sum(S**2,axis=1)

# get all atom participations for full
def get_all_atom_participations(S):
    new_shape=(int(S.shape[0]/3),3,S.shape[1])
    P=np.sum(S.reshape(new_shape)**2,axis=1)
    return P

# get effective masses from an array of masses and full participations
def get_effective_masses(masses,P):
    ems=np.matmul(masses.reshape(1,masses.shape[0]),P)
    return ems.flatten()

# get angular_frequencies form spring constants and effective get_masses
def get_angular_freq(k,em):
    omega=np.sqrt(k/em)*1e13 # omega in rads/s
    return omega

#convert angular frequency to frequency
def convert_angular_freq_to_freq(omega):
    nu=omega/(2*np.pi)  # nu in Hz
    return nu

#get period from frequency
def get_period_from_frequency(nu):
    T=1.0/nu # period in secods
    return T

# normalize and get scaled values
def get_coloring(P):
    m = np.max(P)
    return np.cbrt(P/m)

# Will retunr in KJ/mol/Angtrom^2 assuming D is in Angtrom^2
def spring_constants_from_variances(D,T=293.1):
    R=8.3145
    return R*T/D

# calculate effective masses and derived stats
def calc_em_and_derived_stats(masses,P,k):
    ems=get_effective_masses(masses,P)
    omegas=get_angular_freq(k,ems)
    nus=convert_angular_freq_to_freq(omegas)
    Ts=get_period_from_frequency(nus)
    return ems, omegas, nus, Ts

# mode eignevector should be in nx3 form
def make_color_column(S):
    P=get_atom_participation_from_eigenvector(S)
    B=get_coloring(P)
    return pd.DataFrame(data=B,columns=color_items)

def shift_by_mode(df,mode,indeces,mul):
    stat_items=['x_coord', 'y_coord', 'z_coord']
    to_change = df.iloc[match_col_in_int_list(df,'atom_number',indeces)]
    coords = to_change[stat_items]
    coords += mode*float(mul)
    to_ret = to_change.copy()
    to_ret[stat_items] = coords
    to_ret[color_items] = make_color_column(mode)
    return to_ret

def get_movie_muls(mul,movie_steps):
    ts=np.linspace(-float(0),float(movie_steps),num=movie_steps+1)
    muls=float(mul)*np.cos(2*np.pi*ts/movie_steps)
    return muls

def make_movie_from_muls(muls,ndxfile,pdbfile,mode,newpdbfile,ndx_name):
    ndx=read_ndx(ndxfile)
    print(ndx[ndx_name])
    ppdb=PandasPdb()
    ppdb.read_pdb(pdbfile)
    for i in range(len(muls)):
        new_df = shift_by_mode(ppdb.df['ATOM'],mode,ndx[ndx_name],muls[i])
        mode_pdb = PandasPdb()
        mode_pdb.df['ATOM'] = new_df
        mode_pdb.to_pdb(path=newpdbfile+str(i)+'.pdb',\
            records=['ATOM'], gz=False, append_newline=True)

def modes(xvgfile,ndxfile,pdbfile,mode_idx,newpdbfile,mul,\
    fit_using_pdb=True,ndx_name='C-alpha',movie_steps=200):
    fitfile = pdbfile if fit_using_pdb else None
    mean1, mean2, cov, s, u, v = get_xvg_stats(xvgfile,fitfile=fitfile)
    shift_shape = (int(u.shape[1]/3),3)
    mode = u[:,int(mode_idx)].reshape(shift_shape)
    print("u[:,",mode_idx,"] =",mode)
    muls=get_movie_muls(mul,movie_steps)
    make_movie_from_muls(muls,ndxfile,pdbfile,mode,newpdbfile,ndx_name)
