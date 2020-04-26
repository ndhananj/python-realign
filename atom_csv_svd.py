################################################################################
# get the svd of a csv created from a pdb
# Originally made by Nithin Dhananjayan (ndhanananj@ucdavis.edu)
# Usage : python <this_file_name> <atom_csv file>
# example : python atom_csv_svd dimer1.csv
################################################################################
from alignment import *

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    mean1, mean2, cov, s, u, v, df1, df2 = calc_stats(df,df)
    print_stats(mean1, cov, s, u, v)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_coords(ax, mean1, s, u, df1)
    plt.show()
