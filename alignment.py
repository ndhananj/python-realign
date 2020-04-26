################################################################################
# Originally made by Nithin Dhananjayan (ndhanananj@ucdavis.edu)
# Module to keep alignment related functions
################################################################################

import pandas as pd
import numpy as np
from numpy.linalg import svd
import sys

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

def calc_stats(df1,df2,stat_items=['X','Y','Z']):
    coords1 = df1.filter(items=stat_items).to_numpy()
    coords2 = df2.filter(items=stat_items).to_numpy()
    mean1 = coords1.mean(axis=0)
    mean2 = coords2.mean(axis=0)
    cov = np.matmul((coords2-mean2).T,coords1-mean1)/coords1.shape[0]
    u, s, vh = svd(cov)
    return mean1, mean2, cov, s, u, vh.T, df1, df2

def print_stats(mean, cov, s, u, v):
    print("The mean of coordinates is : ", mean)
    print("The covariance of the coordinates is : \n", cov)
    print("The singular values of that are : ",s)
    print("The left singular vectors are : \n", u)
    print("The right singular vectors are : \n", v)

def svd_ellipsoid(mean,s,u):
    phi = np.linspace(0,2*np.pi, 256).reshape(256, 1) # the angle of the projection in the xy-plane
    theta = np.linspace(0, np.pi, 256).reshape(-1, 256) # the angle from the polar axis, ie the polar angle
    x=np.sin(theta)*np.cos(phi)*np.sqrt(s[0])
    y=np.sin(theta)*np.sin(phi)*np.sqrt(s[1])
    z=np.cos(theta)*np.ones(phi.shape)*np.sqrt(s[2])
    scaled = np.stack([x, y, z])
    dims = scaled.shape
    r=scaled.transpose(1,2,0).reshape([dims[1], dims[2], dims[0], 1])
    tilted = np.matmul(u,r)
    retVal=np.add(tilted,mean.reshape(dims[0],1)).transpose(2,0,1,3).reshape(dims)
    return retVal

def plot_coords(ax, mean, s, u, df, stat_items=['X','Y','Z'],point_color='b', stats_color='r', points_alpha=0.05, stats_alpha=0.05, show_stats=True, show_points=True):
    coords = df.filter(items=stat_items).to_numpy()
    if(show_points):
        ax.scatter(coords[:,0], coords[:,1], coords[:,2], color=point_color, alpha=points_alpha)
    if(show_stats):
        ax.scatter(mean[0], mean[1], mean[2], color=stats_color, alpha=1)
        ellipsoid = svd_ellipsoid(mean, s, u)
        ax.plot_wireframe(ellipsoid[0], ellipsoid[1], ellipsoid[2],  rstride=4, cstride=4, color=stats_color, alpha=stats_alpha)
    ax.set_xlabel(stat_items[0])
    ax.set_ylabel(stat_items[1])
    ax.set_zlabel(stat_items[2])

def find_align(src_df,trg_df, stat_items=['X','Y','Z']):
    src_mu,trg_mu,cov,s,u,v,src_df1,trg_df1=calc_stats(src_df,trg_df, stat_items=stat_items)
    rot_mat = np.matmul(u,v.transpose())
    det = np.linalg.det(rot_mat)
    print("Rotation determinant is : ", det)
    if(det<0):
        rot_mat =  np.matmul(np.matmul(u,np.diag([1,1,-1])),v.transpose())
    return src_mu, trg_mu, rot_mat

def realign(src_df,src_mu, trg_mu, rot_mat, stat_items=['X','Y','Z']):
    coords = src_df.filter(items=stat_items).to_numpy()
    coords -= src_mu
    coords = np.matmul(coords,rot_mat.T)
    coords += trg_mu
    df_coords =  pd.DataFrame(coords,columns=stat_items)
    alg_df = src_df.copy()
    alg_df[stat_items]=df_coords[stat_items]
    return alg_df, rot_mat

def align_df(src_df,trg_df, stat_items=['X','Y','Z']):
    src_mu, trg_mu, rot_mat = find_align(src_df,trg_df, stat_items=stat_items)
    return realign(src_df,src_mu, trg_mu, rot_mat, stat_items=stat_items)
