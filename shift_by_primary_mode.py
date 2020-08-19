from modes import *

import sys

if __name__ == '__main__':
    mn = [i for i in range(int(sys.argv[4]),int(sys.argv[5])+1)]
    if len(sys.argv)>8:
        modes(sys.argv[1],sys.argv[2],sys.argv[3], mn,\
           sys.argv[6],sys.argv[7],ndx_name=sys.argv[8])
    else:
        modes(sys.argv[1],sys.argv[2],sys.argv[3],mn,\
           sys.argv[6],sys.argv[7])
