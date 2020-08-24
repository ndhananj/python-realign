

from modes import *

import sys

def get_mode_stats(mode_idx_beg=0,mode_idx_end=9,\
    eigenmatrix='eigenmatrix.npy',\
    eigenvalues='eigenvalues.npy',\
    spring_constants='spring_constants.npy',\
    effective_masses='effective_masses.npy',\
    residues='residues.npy',\
    mode_stats_file='mode_stats.csv'):
    S=load_matrix(eigenmatrix)
    D=load_matrix(eigenvalues)
    k=load_matrix(spring_constants)
    em=load_matrix(effective_masses)
    resi=load_matrix(residues)
    S_to_process = S[:,mode_idx_beg:mode_idx_end+1]
    D_to_save = D[mode_idx_beg:mode_idx_end+1]
    new_shape = (int(S_to_process.shape[0]/3),3,S_to_process.shape[1])
    S_to_save = S_to_process.reshape(new_shape)
    P_to_process = get_all_atom_participations(S_to_process)
    C_to_save = get_all_colorings(P_to_process,resi)
    k_to_process = k[mode_idx_beg:mode_idx_end+1]
    em_to_process = em[mode_idx_beg:mode_idx_end+1]
    omegas, nus, Ts = calc_em_k_derived_stats(k_to_process,em_to_process)
    labels = ["eigenvalues (A^2)",
        "spring constants (KJ/mol/A^2)",
        "effective masses (g/mol)",
        "angular frequencies (rads/s)",
        "frequencies (Hz)",
        "periods (s)"]
    mode_stat_values = np.stack([
        D_to_save,
        k_to_process,
        em_to_process,
        omegas,
        nus,
        Ts]).T
    mode_stats=pd.DataFrame(data=mode_stat_values,columns=labels)
    mode_stats.to_csv(mode_stats_file)
    for mode_idx in range(mode_idx_beg,mode_idx_end+1):
        save_matrix('eigenvector'+str(mode_idx)+'.npy',S_to_process[:,mode_idx])
        save_matrix('participation'+str(mode_idx)+'.npy',P_to_process[:,mode_idx])
        save_matrix('colors'+str(mode_idx)+'.npy',C_to_save[:,mode_idx])

if __name__ == '__main__':
    # range of modes
    mode_idx_beg = int(sys.argv[1]) if len(sys.argv)>1 else 0
    mode_idx_end = int(sys.argv[2]) if len(sys.argv)>2 else 9
    # data inputs
    eigenmatrix = sys.argv[3] if len(sys.argv)>3 \
       else 'eigenmatrix.npy'
    eigenvalues = sys.argv[4] if len(sys.argv)>4 \
       else 'eigenvalues.npy'
    spring_constants = sys.argv[5] if len(sys.argv)>5 \
       else 'spring_constants.npy'
    effective_masses = sys.argv[6] if len(sys.argv)>6 \
       else 'effective_masses.npy'
    residues = sys.argv[7] if len(sys.argv)>7 \
       else 'residues.npy'
    # output file
    mode_stats_file = sys.argv[8] if len(sys.argv)>8 \
       else 'mode_stats_file.csv'
    get_mode_stats(mode_idx_beg,mode_idx_end,eigenmatrix,eigenvalues,\
       spring_constants,effective_masses,residues,mode_stats_file)
