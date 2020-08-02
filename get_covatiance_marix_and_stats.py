

from modes import *

import sys

if __name__ == '__main__':
    unbias = sys.argv[2] if len(sys.argv)>2 else False
    mean1, mean2, cov, s, u, v, df1, df2 = get_xvg_stats(sys.argv[1],unbias=unbias)
    np.set_printoptions(threshold=sys.maxsize)
    np.core.arrayprint._line_width = 1800
    print("mean ",mean1)
    print("cov ", cov.shape)
    print("s ", s.shape)
    print("u ", u.shape)
    print("v ",v.shape)
    print("sum(u-v)=",np.sum(u-v))
    print("s = ", s)
    shift_shape = (int(u.shape[1]/3),3)
    primary_mode = u[:,0].reshape(shift_shape)
    print("u[:,0] =",primary_mode)
