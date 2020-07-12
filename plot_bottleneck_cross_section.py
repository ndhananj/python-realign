################################################################################
# Plot Cross section of bottleneck
# Originally made by Nithin Dhananjayan (ndhanananj@ucdavis.edu)
# Usage : python <this_file_name> <bottleneck_file>
# example : python plot_bottleneck_cross_section.py bottleneck.pdb
################################################################################
from projections import *
from biopandas.pdb import PandasPdb

if __name__ == "__main__":
    df, radii, n, mean, coords, coords_u, \
        coords_s, coords_vh, proj_xy, plot_df, plot_range, plot_img = \
        proj_stats(sys.argv[1])
    print(plot_range)
    print(coords_s)
    print(coords_s/np.sqrt(n))
    print(plot_img.shape)
    (neg_fill, centers) = get_negative_fill(plot_img)
    print(len(centers))
    fig = plt.figure()
    ax2 = fig.add_subplot(211)
    ax2.spy(plot_img)
    ax2 = fig.add_subplot(212)
    ax2.spy(neg_fill)
    plt.show()
