import pandas as pd
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
import trimesh
import matplotlib.pyplot as plt
import networkx as nx
import os

try:
    import params
except ImportError:
    import params_default as params


def read_twist_x(fname='twisted_x.npy', val=10, limits=[-100,100]):

    if os.path.isfile(fname):
        return np.load(fname)
    else:
        parab = lambda x: 50-x*x/10
        data = []
        size = 25
        val_range = range(limits[0], limits[-1]+1)
        for x in val_range:
            data += list(np.random.multivariate_normal(mean=[x,parab(x),0], cov=[[val,0,0],[0,val,0],[0,0,val]], size=size))#list(np.random.multivariate_normal(mean=[x,x,x], cov=[[val,0,0],[0,val,0],[0,0,val]], size=50))

        for x in val_range:
            data += list(np.random.multivariate_normal(mean=[0,-parab(x),x], cov=[[val,0,0],[0,val,0],[0,0,val]], size=size))#list(np.random.multivariate_normal(mean=[x,x,x], cov=[[val,0,0],[0,val,0],[0,0,val]], size=50))

        np.save(fname, data)
        return np.array(data)




def read_3d_obj(fname="lion_data.npy", fname_obj="lion-reference.obj"):

    if os.path.isfile(fname):
        return np.load(fname)
    else:

        mesh = trimesh.load_mesh(fname_obj)
        a = np.array(mesh.vertices)
        x,y,z = -a[:,0],a[:,2],a[:,1]
        data = np.array(list(zip(x,y,z)))
        
        np.save(fname, data)

        return data



def plot_3d(data, fname='3d.png'):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x,y,z = zip(*data)
    
    ax.scatter(x,y,z, s=20, c='k', alpha=0.6)

    ax.view_init(params.ANGLE[0], params.ANGLE[1])
    fig.savefig(fname, format='png')
    plt.close()




def plot_clustering_3d(obj, data_local, data_global, filename):


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = get_colors(len(obj.c_to_ind))

    ax.plot(*zip(*data_global), marker='o', color='k', ls='', ms=4., mew=1.0, alpha=0.4, mec='none')

    for i,c in enumerate(obj.c_to_ind):
        ax.plot(*zip(*data_local[obj.c_to_ind[c]]), marker='o', color=colors[i], ls='', ms=4., mew=1.0, alpha=0.8, mec='none')
        

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.view_init(*params.ANGLE)
    #plt.grid(True)
    #ax.set_axis_bgcolor('grey')


    
    fig.savefig(filename, format='png')

    plt.close()




def make_spirals():

    r_range = np.array([2**i for i in range(0,3)])


    def polar_to_cart(r, the):
        return (r*np.cos(the), r*np.sin(the))

    def gen_spiral(shift=0, orient=0):
        result = []

        for i, r in enumerate(r_range):
            step = .05/r
            the_range = np.arange(0, -np.pi, -step) if (i+orient)%2!=0 else np.arange(np.pi, 0, -step)
            
            for the in the_range:
                x, y = polar_to_cart(r, the)
                x += shift

                result.append( (x,y) )
            
            shift += r if (i+orient)%2!=0 else -r
        return result

    


    total = gen_spiral(shift=-1, orient=1) + gen_spiral()

    result = np.array(total)

    fig = plt.figure(figsize=(10, 10))

    plt.plot(result[:,0], result[:,1], 'ro', color='k',)

    

    fig.savefig('cosa.png', format='png')

    plt.close()


    result = []

    z = np.random.normal


    for v in total:
        result.append([v[0], v[1], z(0, .1)])

    np.save('spirals.npy', np.array(result))






    



    



    




