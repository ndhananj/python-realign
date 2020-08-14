

from modes import *

import sys

if __name__ == '__main__':
    unbias = sys.argv[5] if len(sys.argv)>5 else False
    pdbToAlign = sys.argv[4] if len(sys.argv)>4 else None
    mode_idx_end = int(sys.argv[3]) if len(sys.argv)>3 else 9
    mode_idx_beg = int(sys.argv[2]) if len(sys.argv)>2 else 0
    mean1, mean2, cov, D, S, S2 = \
        get_xvg_stats(sys.argv[1],fitfile=pdbToAlign,unbias=unbias)
    if pdbToAlign:
        pdb=PandasPdb()
        pdb.read_pdb(pdbToAlign)
        masses=get_masses_from_pdb_by_resn(pdb)
    else :
        masses=None
    np.set_printoptions(threshold=sys.maxsize)
    np.core.arrayprint._line_width = 1800
    #print("mean ",mean1)
    print("cov ", cov.shape)
    save_matrix('covariance.npy',cov)
    print("D ", D.shape)
    print("S ", S.shape)
    print("S2 ",S2.shape)
    err = np.abs(S-S2)
    print("max|S-S2|=",np.max(err))
    print("mean(err)=",np.mean(err))
    #print("D = ", D)
    save_matrix('eigenvalues.npy',D)
    k=spring_constants_from_variances(D)
    save_matrix('spring_constants.npy',k)
    save_matrix('eigenmatrix.npy',S)
    P=get_all_atom_participations(S)
    save_matrix('participations.npy',P)
    if None is not masses:
        ems=get_effective_masses(masses,P)
        save_matrix('effective_masses.npy',ems)
        omegas=get_angular_freq(k,ems)
        save_matrix('angular_frequencies.npy',omegas)
        nus=convert_angular_freq_to_freq(omegas)
        save_matrix('frequencies.npy',nus)
        periods=get_period_from_frequency(nus)
        save_matrix('periods.npy',periods)
    shift_shape = (int(S.shape[1]/3),3)
    k_strings = []
    m_strings = []
    omega_strings = []
    nu_strings = []
    T_strings = []
    for mode_idx in range(mode_idx_beg,mode_idx_end+1):
        mode = S[:,mode_idx].reshape(shift_shape)
        #print("S[:,"+str(mode_idx)+"] =",mode)
        save_matrix('eigenvector'+str(mode_idx)+'.npy',mode)
        P = get_atom_participation_from_eigenvector(mode)
        save_matrix('participation'+str(mode_idx)+'.npy',P)
        B = get_coloring(P)
        save_matrix('colors'+str(mode_idx)+'.npy',B)
        if None is not masses:
            em=P.dot(masses)
            omega=np.sqrt(k[mode_idx]/em)*1e13 # omega in rads/s
            nu=omega/(2*np.pi) # nu in Hz
            T=1.0/nu # period in secods
            k_strings.append(f"{k[mode_idx]:.2e} KJ/mol/A^2")
            m_strings.append(f"{em:.2e} g/mol")
            omega_strings.append(f"{omega:.2e} rads/s")
            nu_strings.append(f"{nu:.2e} Hz")
            T_strings.append(f"{T:.2e} s")
    if None is not masses:
        print("spring constants:",k_strings)
        print("effective masses:",m_strings)
        print("omegas:",omega_strings)
        print("nus:",nu_strings)
        print("Periods:",T_strings)
