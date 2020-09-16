
from gmx_file_processing import *
from alignment import *


import matplotlib.pyplot as plt

stat_items=['x_coord', 'y_coord', 'z_coord']

color_items=['b_factor']

resi_tems=['residue_number']

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

def get_resi_from_atoms(atoms):
    return atoms[resi_tems].to_numpy().astype(int).flatten()

def get_resi_from_df(df):
    return get_resi_from_atoms(df['ATOM'])

def get_resi(pdb):
    return get_resi_from_df(pdb.df)

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

def get_xvg_coords(xvgfile):
    return get_xvg_data_array_from_file(xvgfile)[:,1:]*10 #nm to A

def get_fitted_coords(coords,trg_c,unbias=False):
    src_cs_shape = (coords.shape[0],int(coords.shape[1]/3),3)
    src_cs = coords.reshape(src_cs_shape)
    print("New shape of data:",src_cs.shape)
    for i in range(src_cs_shape[0]):
        src_mu, trg_mu, rot_mat = find_coords_align(src_cs[i],trg_c,\
            unbias=False,force_mirror=False,force_no_mirror=False)
        src_cs[i] = realign_coords(src_cs[i],src_mu, trg_mu, rot_mat)
    coords = src_cs.reshape((coords.shape[0],coords.shape[1]))
    print("New coordinates calculated")
    return coords

def get_xvg_stats(xvgfile,fitfile=None,unbias=False):
    coords=get_xvg_coords(xvgfile)
    print("Shape of data:",coords.shape)
    if(fitfile):
        print("Fitting...")
        pdb = PandasPdb()
        pdb.read_pdb(fitfile)
        print("Fit file read in")
        trg_c = pdb.df['ATOM'].filter(items=stat_items).to_numpy()
        get_fitted_coords(coords,trg_c,unbias=unbias)
    print("Calculating stats...")
    mean, cov, s, u, v = calc_single_coord_stats(coords,unbias=unbias)
    return mean, cov, s, u, v, coords

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
def get_coloring(P,resi):
    rP = np.zeros(np.max(resi)+1)
    num_res = resi.shape[0]
    for i in range(num_res):
        rP[resi[i]]+=P[i]
    m = np.max(rP)
    rP = np.cbrt(rP/m)
    return [rP[resi[i]] for i in range(num_res)]

# get all the colorings for each column
def get_all_colorings(P,resi):
    f = lambda x : get_coloring(x,resi)
    return np.apply_along_axis(f, 0, P)

# Will retunr in KJ/mol/Angtrom^2 assuming D is in Angtrom^2
def spring_constants_from_variances(D,T=293.1):
    R=8.3145e-3 #KJ/mol/Kelvin
    return R*T/D

# calculate effective mass spring constant derived stats
def calc_em_k_derived_stats(k,ems):
    omegas=get_angular_freq(k,ems)
    nus=convert_angular_freq_to_freq(omegas)
    Ts=get_period_from_frequency(nus)
    return omegas, nus, Ts

# calculate effective masses and derived stats
def calc_em_and_derived_stats(masses,P,k):
    ems=get_effective_masses(masses,P)
    omegas, nus, Ts = calc_em_k_derived_stats(k,ems)
    return ems, omegas, nus, Ts

# mode eignevector should be in nx3 form
def make_color_column(S,resi):
    P=get_atom_participation_from_eigenvector(S)
    B=get_coloring(P,resi)
    return pd.DataFrame(data=B,columns=color_items)

def shift_by_mode(df,mode,indeces,mul):
    stat_items=['x_coord', 'y_coord', 'z_coord']
    to_change = df.iloc[match_col_in_int_list(df,'atom_number',indeces)]
    resi = get_resi_from_atoms(df)
    coords = to_change[stat_items]
    coords += mode*float(mul)
    to_ret = to_change.copy()
    to_ret[stat_items] = coords
    to_ret[color_items] = make_color_column(mode,resi)
    return to_ret

# combine individual frames into a complete pdb movie
def concatenate_pdbs(pdb_frame, frame_index, movie_pdb_file):
    # concatenation loop
    movie_pdb_file.write("TITLE    frame t= " + str(frame_index) +"\n")
    movie_pdb_file.write("MODEL    1\n")
    with open(pdb_frame, 'r') as f:
        movie_pdb_file.write(f.read())
    movie_pdb_file.write("TER\n")
    movie_pdb_file.write("ENDMDL\n")

# get multiplier for a movie
def get_movie_muls(mul,movie_steps):
    ts=np.linspace(-float(0),float(movie_steps),num=movie_steps+1)
    muls=float(mul)*np.cos(2*np.pi*ts/movie_steps)
    return muls

def make_movie_from_muls(muls,ndxfile,pdbfile,mode,newpdbfile,ndx_name):
    ndx=read_ndx(ndxfile)
    #print(ndx[ndx_name])
    ppdb=PandasPdb()
    ppdb.read_pdb(pdbfile)
    movie_pdb_file = newpdbfile+'.pdb'
    movie = open(movie_pdb_file, 'w')
    for i in range(len(muls)):
        new_df = shift_by_mode(ppdb.df['ATOM'],mode,ndx[ndx_name],muls[i])
        mode_pdb = PandasPdb()
        mode_pdb.df['ATOM'] = new_df
        pdb_frame = newpdbfile+"_"+str(i)+'.pdb'
        mode_pdb.to_pdb(path=pdb_frame,\
            records=['ATOM'], gz=False, append_newline=True)
        concatenate_pdbs(pdb_frame, i, movie)
        os.remove(pdb_frame)
    movie.close()

def create_mode_movie(mul,movie_steps,ndxfile,pdbfile,mode,newpdbfile,ndx_name):
    muls=get_movie_muls(mul,movie_steps)
    make_movie_from_muls(muls,ndxfile,pdbfile,mode,newpdbfile,ndx_name)

def modes(xvgfile,ndxfile,pdbfile,mode_indices,newpdbfile,mul,\
    fit_using_pdb=True,ndx_name='System',movie_steps=150):
    fitfile = pdbfile if fit_using_pdb else None
    mean1, mean2, cov, s, u, v, coords = get_xvg_stats(xvgfile,fitfile=fitfile)
    shift_shape = (int(u.shape[1]/3),3)
    for mode_idx in mode_indices:
        mode_pdb_file=newpdbfile+"_mode"+str(mode_idx)
        mode = u[:,int(mode_idx)].reshape(shift_shape)
        #print("u[:,",mode_idx,"] =",mode)
        create_mode_movie(mul,movie_steps,\
            ndxfile,pdbfile,mode,mode_pdb_file,ndx_name)

# get the involvement of specific residues in a mode based on participation
def involvement_in_mode_based_on_participation(P,resi,toInclude):
    num_res=resi.shape[0]
    I=np.sum([P[i,:] for i in range(num_res) if resi[i] in toInclude],axis=0)
    return I

# plot involvemenet
def plot_involvement(I,involvement_string="Involvement",mode_end=40):
    fig= plt.figure()
    ax = fig.add_subplot(111)
    mode_end = I.shape[0] if None==mode_end else mode_end
    ax.bar(range(mode_end),I[:mode_end])
    ax.set_xlabel('Mode')
    ax.set_ylabel(involvement_string)
    plt.ylim(I[:mode_end].min()*1.1,I[:mode_end].max()*1.1)
    plt.savefig(involvement_string+".jpg")
    #plt.show()

# calculate surface area of triangles made from centroid
def calc_triangulated_surface_area(xvgfile):
    coords=get_xvg_coords(xvgfile)
    cs_shape = (coords.shape[0],int(coords.shape[1]/3),3)
    cs = coords.reshape(cs_shape)
    centroid = np.mean(cs,axis=1)
    vecs = np.array([cs[i]-centroid[i] for i in range(cs.shape[0]) ])
    all_areas= np.array([np.linalg.norm(np.cross(vecs[:,i-1],vecs[:,i]),axis=1) \
       for i in range(1,vecs.shape[1])])
    areas=np.sum(all_areas,axis=0)
    return 0.5*areas

# plot areas
def plot_area(A,time_step=0.05,area_string="Area"):
    N=A.shape[0]
    T=N*time_step
    times=np.linspace(0,T,num=N)
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(times,A)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel(area_string)
    plt.ylim(A.min()*0.9,A.max()*1.1)
    plt.savefig(area_string+".jpg")
    #plt.show()
