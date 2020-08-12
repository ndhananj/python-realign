
from modes import *

if __name__ == '__main__':
    m=load_matrix(sys.argv[1])
    np.set_printoptions(threshold=sys.maxsize)
    print(m)
