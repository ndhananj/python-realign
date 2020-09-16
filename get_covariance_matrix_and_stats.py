

from modes import *

import sys

def get_covariance_matrix_and_stats(xvg_input,\
    pdbToAlign=None, \
    unbias=False, \
    coordinate='coordinates.npy',\
    covariance='covariance.npy', \
    eigenvalues='eigenvalues.npy', \
    eigenmatrix='eigenmatrix.npy', \
    participations='participations.npy'):
    mean, cov, D, S, S2, coords = \
        get_xvg_stats(xvg_input,fitfile=pdbToAlign,unbias=unbias)
    err = np.abs(S-S2)
    k=spring_constants_from_variances(D)
    P=get_all_atom_participations(S)
    print("coords ", coords.shape)
    print("cov ", cov.shape)
    print("D ", D.shape)
    print("S ", S.shape)
    print("S2 ",S2.shape)
    print("max|S-S2|=",np.max(err))
    print("mean(err)=",np.mean(err))
    save_matrix(coordinates,coords)
    save_matrix(covariance,cov)
    save_matrix(eigenvalues,D)
    save_matrix(spring_constants,k)
    save_matrix(eigenmatrix,S)
    save_matrix(participations,P)

if __name__ == '__main__':
    #inputs
    xvg_input = sys.argv[1]
    pdbToAlign = sys.argv[2] if len(sys.argv)>2 else None
    #parameters
    unbias = sys.argv[3] if len(sys.argv)>3 else False
    #outputs
    coordinates = sys.arg[4] if len(sys.argv)>4 \
        else 'coordinates.npy'
    covariance = sys.arg[5] if len(sys.argv)>5 \
        else 'covariance.npy'
    eigenvalues = sys.arg[6] if len(sys.argv)>6 \
        else 'eigenvalues.npy'
    spring_constants = sys.argv[7] if len(sys.argv)> 7 \
       else 'spring_constants.npy'
    eigenmatrix = sys.argv[8] if len(sys.argv)>8 \
       else 'eigenmatrix.npy'
    participations = sys.argv[9] if len(sys.argv)>9 \
       else 'participations.npy'
    # calculate
    get_covariance_matrix_and_stats(xvg_input,\
        pdbToAlign, \
        unbias, \
        coordinates, \
        covariance, \
        eigenvalues, \
        eigenmatrix, \
        participations)
