################################################################################
# allign one csv created from a pdb to another
# Originally made by Nithin Dhananjayan (ndhanananj@ucdavis.edu)
# Usage : python <this_file_name> <source_atom_csv file> <target_atom_csv file> <output file>
# example : python align_atom_csv.py dimer2.csv dimer1.csv dimer2_aligned.csv
################################################################################

from alignment import *

if __name__ == "__main__":
    src_df = pd.read_csv(sys.argv[1])
    trg_df = pd.read_csv(sys.argv[2])
    src_mu1, src_mu2, src_cov, src_s, src_u, src_v, src_df1, src_df2 = calc_stats(src_df,src_df)
    trg_mu1, trg_mu2, trg_cov, trg_s, trg_u, trg_v, trg_df1, src_df2 = calc_stats(trg_df,trg_df)

    aligned_df, rot_mat = align_df(src_df,trg_df)
    alg_mu1, alg_mu2, alg_cov, alg_s, alg_u, alg_v, alg_df1, alg_df2 = calc_stats(aligned_df,aligned_df)
    aligned_df.to_csv(sys.argv[3],index=False)

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
    plot_coords(ax, src_mu1, src_s, src_u, src_df1, show_points=False, point_color='b', stats_color='r', points_alpha=0.01)
    plot_coords(ax, trg_mu1, trg_s, trg_u, trg_df1, show_points=False, point_color='g', stats_color='k', points_alpha=0.01)
    plot_coords(ax, alg_mu1, alg_s, alg_u, alg_df1, show_points=False, point_color='c', stats_color='m', points_alpha=0.01)
    plt.show()
