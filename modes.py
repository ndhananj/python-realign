
from gmx_file_processing import *
from alignment import *


import matplotlib.pyplot as plt

def get_xvg_stats(xvgfile,unbias=False):
    xvg=read_xvg(xvgfile)
    df = xvg['data']
    mean1, mean2, cov, s, u, v, df1, df2 = \
       calc_stats(df,df,stat_items=xvg['yaxis labels'],unbias=unbias)
    return mean1, mean2, cov, s, u, v, df1, df2

def shift_by_mode(df,primary_mode,indeces,mul):
    stat_items=['x_coord', 'y_coord', 'z_coord']
    to_change = df.iloc[match_col_in_int_list(df,'atom_number',indeces)]
    coords = to_change[stat_items]
    coords += primary_mode*float(mul)
    to_ret = to_change.copy()
    to_ret[stat_items] = coords
    return to_ret

def modes(xvgfile,ndxfile,pdbfile,mode,newpdbfile,mul):
    mean1, mean2, cov, s, u, v, df1, df2 = get_xvg_stats(xvgfile)
    shift_shape = (int(u.shape[1]/3),3)
    primary_mode = u[:,int(mode)].reshape(shift_shape)
    print("u[:,",mode,"] =",primary_mode)
    ndx=read_ndx(ndxfile)
    print(ndx['C-alpha'])
    ppdb=PandasPdb()
    ppdb.read_pdb(pdbfile)
    ts=np.linspace(-float(0),float(400),num=401)
    muls=float(mul)*np.cos(np.pi*ts/200)
    for i in range(len(muls)):
        new_df = shift_by_mode(ppdb.df['ATOM'],primary_mode,ndx['C-alpha'],muls[i])
        mode_pdb = PandasPdb()
        mode_pdb.df['ATOM'] = new_df
        mode_pdb.to_pdb(path=newpdbfile+str(i)+'.pdb',\
            records=['ATOM'], gz=False, append_newline=True)
