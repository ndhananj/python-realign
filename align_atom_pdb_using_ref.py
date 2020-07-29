################################################################################
# align one pdb using the rotation needed to align a reference source to a target
# Originally made by Nithin Dhananjayan (ndhanananj@ucdavis.edu)
# Usage : python <this_file_name> <ref_pdb_file> <ref_pdb_file> <src_pdb_file> <ouput_file>
# example : python align_atom_pdb_using_ref.py dimer2.pdb dimer1.pdb dimer2.pdb dimer2_aligned.pdb
################################################################################

from alignment import *
from biopandas.pdb import PandasPdb

if __name__ == "__main__":
    ref_src_pdb = PandasPdb()
    ref_src_pdb.read_pdb(sys.argv[1])
    ref_trg_pdb = PandasPdb()
    ref_trg_pdb.read_pdb(sys.argv[2])
    src_pdb = PandasPdb()
    src_pdb.read_pdb(sys.argv[3])
    force_mirror = bool(sys.argv[4]) if(len(sys.argv)>4) else False
    force_no_mirror = bool(sys.argv[5]) if(len(sys.argv)>5) else False
    print("force_mirror : ", force_mirror, "force_no_mirror : ", force_no_mirror)


    ref_src = ref_src_pdb.df['ATOM']
    ref_trg = ref_trg_pdb.df['ATOM']
    src_df = src_pdb.df['ATOM']
    stat_items=['x_coord', 'y_coord', 'z_coord']
    ref_src_mu1, ref_src_mu2, ref_src_cov, ref_src_s, ref_src_u, ref_src_v, ref_src_df1, ref_src_df2 = calc_stats(ref_src,ref_src,stat_items=stat_items)
    ref_trg_mu1, ref_trg_mu2, ref_trg_cov, ref_trg_s, ref_trg_u, ref_trg_v, ref_trg_df1, ref_src_df2 = calc_stats(ref_trg,ref_trg,stat_items=stat_items)
    src_mu1, src_mu2, src_cov, src_s, src_u, src_v, src_df1, src_df2 = calc_stats(src_df,src_df,stat_items=stat_items)

    aligned_df, rot_mat = align_df_using_ref(ref_src,ref_trg,src_df,stat_items=stat_items,force_mirror=force_mirror,force_no_mirror=force_no_mirror)
    alg_mu1, alg_mu2, alg_cov, alg_s, alg_u, alg_v, alg_df1, alg_df2 = calc_stats(aligned_df,aligned_df,stat_items=stat_items)
    aligned_pdb = PandasPdb()
    aligned_pdb.df['ATOM'] = aligned_df
    aligned_pdb.to_pdb(path=sys.argv[4], records=['ATOM'], gz=False, append_newline=True)

    print("Stats for source :")
    print_stats(ref_src_mu1, ref_src_cov, ref_src_s, ref_src_u, ref_src_v)
    print("Stats for target :")
    print_stats(ref_trg_mu1, ref_trg_cov, ref_trg_s, ref_trg_u, ref_trg_v)
    print("Stats for alignment :")
    print_stats(alg_mu1, alg_cov, alg_s, alg_u, alg_v)
    print("Rotation Matrix :")
    print(rot_mat)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_coords(ax, ref_src_mu1, ref_src_s, ref_src_u, ref_src_df1, stat_items=stat_items, show_points=False, point_color='b', stats_color='r', points_alpha=0.01)
    plot_coords(ax, ref_trg_mu1, ref_trg_s, ref_trg_u, ref_trg_df1, stat_items=stat_items, show_points=False, point_color='g', stats_color='k', points_alpha=0.01)
    plot_coords(ax, alg_mu1, alg_s, alg_u, alg_df1, stat_items=stat_items, show_points=False, point_color='c', stats_color='m', points_alpha=0.01)
    plt.show()
