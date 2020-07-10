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
atom_name = 'atom_name'
vdwr={"C":1.5,"F":1.2,"H":0.4,"N":1.10,"O":1.05,"S":1.6,"":0.0,"NULL":0.0,None:0.0}

def get_vdwr(elem):
    return vdwr[elem] if elem in vdwr else 0.0

def get_elem(df):
    #elems=df[elem_item]   # not always reliably present
    elems=np.array(df[atom_name].astype(str).str.extract("([CFHNOS])")[0])
    return elems

def pdb_atoms(filename):
    bottleneck_pdb = PandasPdb()
    bottleneck_pdb.read_pdb(filename)
    df = bottleneck_pdb.df['ATOM']
    radii = np.array([get_vdwr(i) for i in get_elem(df)])
    coords = df.filter(items=stat_items).to_numpy()
    n = coords.shape[0]
    return df, radii, coords, n

def boolGrid(plot_range, grid_size = 0.01):
    x_size = np.rint((plot_range[1,0]-plot_range[0,0])/grid_size).astype(int)
    print(x_size)
    y_size = np.rint((plot_range[1,1]-plot_range[0,1])/grid_size).astype(int)
    print(y_size)
    x_trans_func = lambda x: (x-plot_range[0,0])/grid_size
    y_trans_func = lambda y: (y-plot_range[0,1])/grid_size
    r_trans_func = lambda r: r/grid_size
    return np.zeros((x_size,y_size),dtype=bool), \
        x_trans_func, y_trans_func, r_trans_func

def plotBool(x,y,r,plot_range):
    plotArr, x_trans_func, y_trans_func, r_trans_func = boolGrid(plot_range)
    (lx,ly) = plotArr.shape
    X, Y = np.ogrid[0:lx, 0:ly]
    N = len(x)
    for n in range(N):
        mask = (X-x_trans_func(x[n]))**2 + (Y-y_trans_func(y[n]))**2 < \
            r_trans_func(r[n])**2
        plotArr[mask] = True
    return np.flipud(plotArr.T)

def overlapPlotImg(smaller_plot_range, larger_plot_range, \
    smaller_plot_img, larger_plot_img):
    plotArr, x_trans_func, y_trans_func, r_trans_func = \
       boolGrid(larger_plot_range)


def get_plot_range_and_img(proj_xy, radii):
    plot_range = \
        np.stack([np.min(proj_xy,axis=0), np.max(proj_xy,axis=0)],axis=0)
    plot_img = plotBool(proj_xy.T[0],proj_xy.T[1],radii, plot_range)
    return plot_range, plot_img

def proj_stats(filename):
    df, radii, coords, n = pdb_atoms(filename)
    mean = np.mean(coords, axis=0)
    coords -= mean
    coords_u, coords_s, coords_vh = svd(coords)
    proj_xy = np.matmul(coords_u[:,0:2],np.diag(coords_s[0:2]))
    plot_range,plot_img = get_plot_range_and_img(proj_xy, radii)
    plot_df = \
        pd.DataFrame({'X':proj_xy.T[0], 'Y':proj_xy.T[1], 'R':radii*1000})
    return df, radii, n, mean, coords, coords_u, \
        coords_s, coords_vh, proj_xy, plot_df, plot_range, plot_img

def within_slice(v,coords_vh,coords_s,n,mean):
    v-=mean
    limits = 3*coords_s/np.sqrt(n)
    projs = np.matmul(v,coords_vh.T)
    lengths = abs(projs)
    return (lengths<limits).all()

def slice_range(my_n, coords, coords_vh, coords_s, n, mean):
    return [i for i in range(my_n) if \
        within_slice(coords[i,:], coords_vh, coords_s, n, mean)]

def slice(filename, coords_vh, coords_s, n, mean):
    df, radii, coords, my_n = pdb_atoms(filename)
    sr = slice_range(my_n, coords, coords_vh, coords_s, n, mean)
    slice_radii = np.array([radii[i] for i in sr ])
    slice_coord = \
        np.matmul(np.array([coords[i] for i in sr]),coords_vh[0:2,:].T)
    plot_range, plot_img = get_plot_range_and_img(slice_coord, radii)
    plot_df = \
        pd.DataFrame({'X':slice_coord.T[0], \
            'Y':slice_coord.T[1], 'R':slice_radii*1000})
    return df, sr, slice_radii, slice_coord, plot_df, plot_range, plot_img

def slice_based_on_pdb(src_file,trg_file):
    trg_df, trg_radii, n, mean, coords, coords_u, coords_s, coords_vh, \
        proj_xy, trg_plot_df, trg_plot_range, trg_plot_img = proj_stats(trg_file)
    src_df, sr, slice_radii, slice_coord, \
        slice_plot_df, slice_plot_range, slice_plot_img = \
        slice(src_file, coords_vh, coords_s, n, mean)
    return slice_plot_df, trg_plot_df, trg_plot_img, slice_plot_img

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
