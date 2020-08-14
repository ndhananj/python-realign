

from modes import *

import sys

def get_mode_stats(mode_idx_beg=0,mode_idx_end=9,\
    eigenmatrix='eigenmatrix.npy',\
    spring_constants='spring_constants.npy',\
    effective_masses='effective_masses.npy'):
    S=load_matrix(eigenmatrix)
    print(S.shape)
    k=load_matrix(spring_constants)
    print(k.shape)
    em=load_matrix(effective_masses)
    print(em.shape)
    k_strings = []
    m_strings = []
    omega_strings = []
    nu_strings = []
    T_strings = []
    shift_shape = (int(S.shape[1]/3),3)
    for mode_idx in range(mode_idx_beg,mode_idx_end+1):
        mode = S[:,mode_idx].reshape(shift_shape)
        save_matrix('eigenvector'+str(mode_idx)+'.npy',mode)
        P = get_atom_participation_from_eigenvector(mode)
        save_matrix('participation'+str(mode_idx)+'.npy',P)
        B = get_coloring(P)
        save_matrix('colors'+str(mode_idx)+'.npy',B)
        omega=np.sqrt(k[mode_idx]/em[mode_idx])*1e13 # omega in rads/s
        nu=omega/(2*np.pi) # nu in Hz
        T=1.0/nu # period in secods
        k_strings.append(f"{k[mode_idx]:.2e} KJ/mol/A^2")
        m_strings.append(f"{em[mode_idx]:.2e} g/mol")
        omega_strings.append(f"{omega:.2e} rads/s")
        nu_strings.append(f"{nu:.2e} Hz")
        T_strings.append(f"{T:.2e} s")
    print("spring constants:\n",k_strings)
    print("effective masses:\n",m_strings)
    print("omegas:\n",omega_strings)
    print("nus:\n",nu_strings)
    print("Periods:\n",T_strings)

if __name__ == '__main__':
    # range of modes
    mode_idx_beg = int(sys.argv[1]) if len(sys.argv)>1 else 0
    mode_idx_end = int(sys.argv[2]) if len(sys.argv)>2 else 9
    # data inputs
    eigenmatrix = sys.argv[3] if len(sys.argv)>3 \
       else 'eigenmatrix.npy'
    spring_constants = sys.argv[4] if len(sys.argv)>4 \
       else 'spring_constants.npy'
    effective_masses = sys.argv[5] if len(sys.argv)>5 \
       else 'effective_masses.npy'
    get_mode_stats(mode_idx_beg,mode_idx_end,\
        eigenmatrix,spring_constants,effective_masses)
