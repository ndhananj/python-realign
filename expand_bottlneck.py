################################################################################
# Expand bottleneck in the plane of its corss section
# Originally made by Nithin Dhananjayan (ndhanananj@ucdavis.edu)
# Usage : python <this_file_name> <bottleneck_file> <expansion> <new_bottlneck>
# example : python expand_bottleneck.py bottleneck.pdb 1.5 expanded_bottleneck.pdb
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
