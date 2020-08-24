

from modes import *

import sys

def get_mass_derived_stats(pdbForMasses,\
    participations='participations.npy',\
    spring_constants='spring_constants.npy',\
    effective_masses='effective_masses.npy',\
    angular_frequencies='angular_frequencies.npy',\
    frequencies='frequencies.npy',\
    periods='periods.npy',\
    residues='residues.npy'):
    # input files
    P=load_matrix(participations)
    k = load_matrix(spring_constants)
    pdb=PandasPdb()
    pdb.read_pdb(pdbForMasses)
    masses=get_masses_from_pdb_by_resn(pdb)
    resi = get_resi(pdb)
    # calculate effective masses and derived stats
    ems, omegas, nus, Ts = calc_em_and_derived_stats(masses,P,k)
    # write out calculated matrices
    save_matrix(effective_masses,ems)
    save_matrix(angular_frequencies,omegas)
    save_matrix(frequencies,nus)
    save_matrix(periods,Ts)
    save_matrix(residues,resi)

if __name__ == '__main__':
    #inputs
    pdbForMasses = sys.argv[1]
    participations = sys.argv[2] if len(sys.argv)> 2 \
        else 'participations.npy'
    spring_constants = sys.argv[3] if len(sys.argv)>3 \
        else 'spring_constants.npy'
    #outputs
    effective_masses = sys.argv[4] if len(sys.argv)> 4 \
        else 'effective_masses.npy'
    angular_frequencies = sys.argv[5] if len(sys.argv)> 5 \
        else 'angular_frequencies.npy'
    frequencies = sys.argv[6] if len(sys.argv)>6 \
        else 'frequencies.npy'
    periods = sys.argv[7] if len(sys.argv)>7 \
        else 'periods.npy'
    residues = sys.argv[8] if len(sys.argv)>8 \
        else 'residues.npy'
    #calculate
    get_mass_derived_stats(pdbForMasses,\
        participations,\
        spring_constants,\
        effective_masses,\
        angular_frequencies,\
        frequencies,\
        periods, \
        residues)
