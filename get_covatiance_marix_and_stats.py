

from modes import *

import sys

if __name__ == '__main__':
    unbias = sys.argv[5] if len(sys.argv)>5 else False
    pdbToAlign = sys.argv[4] if len(sys.argv)>4 else None
    mode_idx_end = int(sys.argv[3]) if len(sys.argv)>3 else 9
    mode_idx_beg = int(sys.argv[2]) if len(sys.argv)>2 else 0
    mean1, mean2, cov, s, u, v = \
        get_xvg_stats(sys.argv[1],fitfile=pdbToAlign,unbias=unbias)
    np.set_printoptions(threshold=sys.maxsize)
    np.core.arrayprint._line_width = 1800
    print("mean ",mean1)
    print("cov ", cov.shape)
    with open('covariance.npy','wb') as f:
        np.save(f,cov)
    print("D ", s.shape)
    print("S ", u.shape)
    print("S2 ",v.shape)
    print("sum(S-S2)=",np.sum(u-v))
    print("D = ", s)
    with open('eigenvalues.npy','wb') as f:
        np.save(f,s)
    shift_shape = (int(u.shape[1]/3),3)
    for mode_idx in range(mode_idx_beg,mode_idx_end+1):
        mode = u[:,mode_idx].reshape(shift_shape)
        print("S[:,"+str(mode_idx)+"] =",mode)
        with open('eigenvector'+str(mode_idx)+'.npy','wb') as f:
            np.save(f,mode)
