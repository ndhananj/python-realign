################################################################################
# Plot Cross section of bottleneck
# Originally made by Nithin Dhananjayan (ndhanananj@ucdavis.edu)
# Usage : python <this_file_name> <bottleneck_file>
# example : python plot_bottleneck_cross_section.py bottleneck.pdb
################################################################################
from projections import *
from biopandas.pdb import PandasPdb

if __name__ == "__main__":
    df, radii, n, mean, coords, coords_u, coords_s, coords_vh, proj_xy, plot_df = proj_stats(sys.argv[1])
    print(coords_s/np.sqrt(n))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter('X', 'Y', s='R', color='b', alpha=1, data=plot_df)
    plt.show()
