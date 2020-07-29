################################################################################
# allign one pdb to another
#  Originally made by Nithin Dhananjayan (ndhanananj@ucdavis.edu)
# Usage : python <this_file_name> <source_atom_pdb file> <target_atom_pdb file> <output file>
# example : python align_atom_pdb.py dimer2.pdb dimer1.pdb dimer2_aligned.pdb
################################################################################

from alignment import *
from biopandas.pdb import PandasPdb

if __name__ == "__main__":
    src_pdb = PandasPdb()
    src_pdb.read_pdb(sys.argv[1])
    trg_pdb = PandasPdb()
    trg_pdb.read_pdb(sys.argv[2])
    force_mirror = bool(sys.argv[3]) if(len(sys.argv)>3) else False
    force_no_mirror = bool(sys.argv[4]) if(len(sys.argv)>4) else False
    print("force_mirror : ", force_mirror, "force_no_mirror : ", force_no_mirror)

    src_df = src_pdb.df['ATOM']
    trg_df = trg_pdb.df['ATOM']
    stat_items=['x_coord', 'y_coord', 'z_coord']
    src_mu1, src_mu2, src_cov, src_s, src_u, src_v, src_df1, src_df2 = calc_stats(src_df,src_df,stat_items=stat_items)
    trg_mu1, trg_mu2, trg_cov, trg_s, trg_u, trg_v, trg_df1, src_df2 = calc_stats(trg_df,trg_df,stat_items=stat_items)

    aligned_df, rot_mat = align_df(src_df,trg_df,stat_items=stat_items,force_mirror=force_mirror,force_no_mirror=force_no_mirror)
    alg_mu1, alg_mu2, alg_cov, alg_s, alg_u, alg_v, alg_df1, alg_df2 = calc_stats(aligned_df,aligned_df,stat_items=stat_items)
    aligned_pdb = PandasPdb()
    aligned_pdb.df['ATOM'] = aligned_df
    aligned_pdb.to_pdb(path=sys.argv[3], records=['ATOM'], gz=False, append_newline=True)

    print("Stats for source :")
    print_stats(src_mu1, src_cov, src_s, src_u, src_v)
    print("Stats for target :")
    print_stats(trg_mu1, trg_cov, trg_s, trg_u, trg_v)
    print("Stats for alignment :")
    print_stats(alg_mu1, alg_cov, alg_s, alg_u, alg_v)
    print("Rotation Matrix :")
    print(rot_mat)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_coords(ax, src_mu1, src_s, src_u, src_df1, stat_items=stat_items, show_points=False, point_color='b', stats_color='r', points_alpha=0.01)
    plot_coords(ax, trg_mu1, trg_s, trg_u, trg_df1, stat_items=stat_items, show_points=False, point_color='g', stats_color='k', points_alpha=0.01)
    plot_coords(ax, alg_mu1, alg_s, alg_u, alg_df1, stat_items=stat_items, show_points=False, point_color='c', stats_color='m', points_alpha=0.01)
    plt.show()
