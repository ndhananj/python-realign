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
from centers import *
from sklearn.decomposition import PCA

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

def adj_con_mask(X,Y,center,con_len):
    xd = abs(X-center[0])/con_len
    valx = np.logical_or(xd==1,xd==0)
    yd = abs(Y-center[1])/con_len
    valy = np.logical_or(yd==1,yd==0)
    return np.logical_and(valx,valy)

def get_con_fill(fill,X,Y,centers,con_len,probe_size,recur=1,count=6):
    con_fill = fill.copy()
    kept_centers=[]
    for center in centers:
        adj_mask = adj_con_mask(X,Y,center,con_len)
        if(np.sum(fill[adj_mask])<count):
            mask = (X-center[0])**2 + (Y-center[1])**2 < probe_size**2
            con_fill[mask] = False
        else:
            kept_centers.append(center)
    return (con_fill, kept_centers) if recur==0 else get_con_fill(\
        con_fill,X,Y,kept_centers,con_len,probe_size,recur=recur-1,count=count)

def dir_fill(img,centers,X,Y,probe_size):
    for center in centers:
        mask = (X-center[0])**2 + (Y-center[1])**2 < probe_size**2
        img[mask] = True
    return img

def get_central_component(con_fill,center,X,Y,con_len,probe_size):
    component_size=0
    central_component={center}
    new_component_size=len(central_component)
    while component_size<new_component_size :
        component_size = new_component_size
        to_add = set()
        for center in central_component:
            adj_mask=adj_con_mask(X,Y,center,con_len)
            idxs = np.argwhere(adj_mask)
            bools = con_fill[adj_mask]
            neighbors={(idxs[i][0],idxs[i][1]) \
                for i in range(len(idxs)) if bools[i]}
            to_add|=neighbors
        central_component.update(to_add)
        new_component_size =len(central_component)
    comp_image = dir_fill(np.zeros(con_fill.shape),\
       central_component,X,Y,probe_size)
    return comp_image, central_component

def get_negative_fill(plot_img,probe_elem='H',grid_size=0.01):
    probe_size = round(vdwr[probe_elem]/grid_size)
    con_len = probe_size*2
    (lx,ly) = plot_img.shape
    X, Y = np.ogrid[0:lx, 0:ly]
    row_range=range(probe_size,lx,con_len)
    col_range=range(probe_size,ly,con_len)
    neg_fill = np.zeros((lx,ly),dtype=bool)
    centers=[]
    for row in row_range:
        for col in col_range:
            mask = (X-row)**2 + (Y-col)**2 < probe_size**2
            if np.logical_not(np.any(plot_img[mask])):
                neg_fill[mask] = True
                centers.append((row,col))
    (neg_fill, centers) = get_con_fill(neg_fill,X,Y,centers,con_len,probe_size)
    return get_central_component(\
       neg_fill,(int(lx/2),int(ly/2)),X,Y,con_len,probe_size)

def get_3D_fill_grid(t_coords,probe_elem,res_thickness=0.0):
    probe_size = vdwr[probe_elem]
    con_len = 2*probe_size
    maxs=np.max(t_coords,axis=0)-(probe_size+res_thickness)
    mins=np.min(t_coords,axis=0)+(probe_size+res_thickness)
    grid=np.mgrid[mins[0]:maxs[0]:con_len,\
       mins[1]:maxs[1]:con_len,mins[2]:maxs[2]:con_len]
    num_centers = grid.shape[1]*grid.shape[2]*grid.shape[3]
    tg_coords=np.stack(grid,axis=-1).reshape((num_centers,3))
    return tg_coords, probe_size, con_len, grid

def num_3D_adj(my_center,bool_grid):
    num=0
    adjacency = range(-1,2)
    for i in adjacency:
        x=my_center[0]+i
        if (x>=0) and (x<bool_grid.shape[0]):
            for j in adjacency:
                y = my_center[1]+j
                if (y>=0) and (y<bool_grid.shape[1]):
                    for k in adjacency:
                        z = my_center[2]+k
                        if (z>=0) and (z<bool_grid.shape[2]):
                            num+=bool_grid[x,y,z]
    return num

def get_3D_con_fill(bool_grid,recur=1,count=20):
    toRet=bool_grid.copy()
    print(len(np.argwhere(bool_grid)))
    for center in np.argwhere(bool_grid):
        if(num_3D_adj(center,bool_grid)<count):
            toRet[center[0],center[1],center[2]]=False
    print(len(np.argwhere(bool_grid)))
    print(len(np.argwhere(toRet)))
    return toRet if recur==0 else get_3D_con_fill(\
        toRet,recur=recur-1,count=count)

def get_3D_cetral_comp(bool_grid): ##### in progress
    grid_center = (bool_grid.shape)/2
    ret_grid = np.zeros(bool_grid.shape).astype(bool)
    component_size=0
    central_component={grid_center}
    new_component_size=len(central_component)
    while component_size<new_component_size :
        component_size = new_component_size
        to_add = set()
        for center in central_component:
            adj_mask=adj_con_mask(X,Y,center,con_len)
            idxs = np.argwhere(adj_mask)
            bools = con_fill[adj_mask]
            neighbors={(idxs[i][0],idxs[i][1]) \
                for i in range(len(idxs)) if bools[i]}
            to_add|=neighbors
        central_component.update(to_add)
        new_component_size =len(central_component)


def grid_to_centers(bool_grid,grid):
    saved=[]
    for center in np.argwhere(bool_grid):
        saved.append([grid[0,center[0],0,0],\
                     grid[1,0,center[1],0],\
                     grid[2,0,0,center[2]]])
    return np.array(saved)

def get_3D_fill_from_grid(coords, radii, g_coords, probe_size, con_len, grid):
    saved = []
    bool_grid = np.zeros(grid.shape[1:]).astype(bool)
    num_coords = coords.shape[0] # should also equal num_radii
    print(bool_grid.shape)
    xdiv = bool_grid.shape[1]*bool_grid.shape[2]
    ydiv = bool_grid.shape[2]
    for j in range(g_coords.shape[0]):
        X = int(j/xdiv)
        Y = int((j%xdiv)/ydiv)
        Z = j%ydiv
        should_save = True
        for i in range(num_coords):
            if np.linalg.norm(g_coords[j]-coords[i])<probe_size+radii[i]:
                should_save = False
                break
        if should_save :
            saved.append(g_coords[j])
            bool_grid[X,Y,Z]=True
    orig_centers = grid_to_centers(bool_grid,grid)
    print(np.sum(np.array(saved)- orig_centers))
    bool_grid=get_3D_con_fill(bool_grid)
    print(np.mean(orig_centers)-np.mean(grid_to_centers(bool_grid,grid)))
    #return np.array(saved)
    return grid_to_centers(bool_grid,grid)

def get_negative_fill_3D_no_output(coords,radii,probe_elem='H'):
    pca=PCA(n_components=3)
    pca.fit(coords)
    t_coords = pca.transform(coords)
    tg_coords, probe_size, con_len, grid = get_3D_fill_grid(t_coords,probe_elem)
    print(tg_coords.shape)
    tf_coords = get_3D_fill_from_grid(\
        t_coords, radii, tg_coords, probe_size, con_len, grid)
    print(tf_coords.shape)
    f_coords = pca.inverse_transform(tf_coords)
    centers = pd.DataFrame(data=f_coords,columns=[10,11,12])
    return centers

def get_negative_fill_3D(filename,out_file,probe_elem='H'):
    df, radii, coords, n = pdb_atoms(filename)
    centers=get_negative_fill_3D_no_output(coords,radii,probe_elem='H')
    output_centers(centers,probe_elem,out_file)
    return coords
