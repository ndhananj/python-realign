################################################################################
# Plot Cross section of bigger file based on bottleneck
# Originally made by Nithin Dhananjayan (ndhanananj@ucdavis.edu)
# Usage : python <this_file_name> <bottleneck_file> <bigger_file>
# example : python plot_cross_section_based_on_bottlneck.py bottleneck.pdb 4hea_atoms.pdb
################################################################################
from projections import *
from biopandas.pdb import PandasPdb

if __name__ == "__main__":
    slice_plot_df, trg_plot_df, trg_plot_img, slice_plot_img = \
        slice_based_on_pdb(sys.argv[2],sys.argv[1])
    fig = plt.figure()
    #ax = fig.add_subplot(211)
    #ax.scatter('X', 'Y', s='R', color='b', alpha=1, data=slice_plot_df)
    #ax.scatter('X', 'Y', s='R', color='r', alpha=0.7, data=trg_plot_df)
    ax2 = fig.add_subplot(111)
    ax2.spy(slice_plot_img)
    ax2.spy(trg_plot_img, alpha=0.01)
    plt.show()
