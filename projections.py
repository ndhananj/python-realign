################################################################################
# Module to do projections
# Originally made by Nithin Dhananjayan (ndhanananj@ucdavis.edu)
################################################################################

import pandas as pd
from biopandas.pdb import PandasPdb
import numpy as np
from numpy.linalg import svd
import sys
import matplotlib.pyplot as plt

stat_items=['x_coord', 'y_coord', 'z_coord']
elem_item = 'element_symbol'
vdwr={"C":1.5,"F":1.2,"H":0.4,"N":1.10,"O":1.05,"S":1.6,"":0.0,"NULL":0.0,None:0.0}

def pdb_atoms(filename):
    bottleneck_pdb = PandasPdb()
    bottleneck_pdb.read_pdb(filename)
    df = bottleneck_pdb.df['ATOM']
    radii = np.array([vdwr[i] for i in df[elem_item]])
    coords = df.filter(items=stat_items).to_numpy()
    n = coords.shape[0]
    return df, radii, coords, n

def proj_stats(filename):
    df, radii, coords, n = pdb_atoms(filename)
    mean = np.mean(coords, axis=0)
    coords -= mean
    coords_u, coords_s, coords_vh = svd(coords)
    proj_xy = np.matmul(coords_u[:,0:2],np.diag(coords_s[0:2]))
    plot_df = pd.DataFrame({'X':proj_xy.T[0], 'Y':proj_xy.T[1], 'R':radii*1000})
    return df, radii, n, mean, coords, coords_u, coords_s, coords_vh, proj_xy, plot_df

def within_slice(v,coords_vh,coords_s,n,mean):
    v-=mean
    limits = 3*coords_s/np.sqrt(n)
    projs = np.matmul(v,coords_vh.T)
    lengths = abs(projs)
    return (lengths<limits).all()

def slice_range(my_n, coords, coords_vh, coords_s, n, mean):
    return [i for i in range(my_n) if within_slice(coords[i,:], coords_vh, coords_s, n, mean)]

def slice(filename, coords_vh, coords_s, n, mean):
    df, radii, coords, my_n = pdb_atoms(filename)
    sr = slice_range(my_n, coords, coords_vh, coords_s, n, mean)
    slice_radii = np.array([radii[i] for i in sr ])
    slice_coord = np.matmul(np.array([coords[i] for i in sr]),coords_vh[0:2,:].T)
    plot_df = pd.DataFrame({'X':slice_coord.T[0], 'Y':slice_coord.T[1], 'R':slice_radii*1000})
    return df, sr, slice_radii, slice_coord, plot_df

def slice_based_on_pdb(src_file,trg_file):
    trg_df, trg_radii, n, mean, coords, coords_u, coords_s, coords_vh, proj_xy, trg_plot_df = proj_stats(trg_file)
    src_df, sr, slice_radii, slice_coord, slice_plot_df = slice(src_file, coords_vh, coords_s, n, mean)
    return slice_plot_df, trg_plot_df

def expanded_bottleneck(src_file,trg_file,factor):
    src_df, src_radii, n, mean, coords, coords_u, coords_s, coords_vh, proj_xy, src_plot_df = proj_stats(src_file)
    expanded_proj = factor*proj_xy
    fat_ep = np.concatenate(expanded_proj,coords_s[2]*coords_u[:,2])
    coords = np.matmul(fat_ep,coords_vh)
    coords += mean
    df_coords =  pd.DataFrame(coords,columns=stat_items)
    trg_df = src_df.copy()
    trg_df[stat_items]=df_coords[stat_items]
    trg_pdb = PandasPdb()
    trg_pdb.df['ATOM'] = trg_df
    trg_pdb.to_pdb(path=trg_file, records=['ATOM'], gz=False, append_newline=True)
