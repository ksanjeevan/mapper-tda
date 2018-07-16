import pandas as pd
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import networkx as nx
import os
import colorsys, errno

try:
    import params
except ImportError:
    import params_default as params


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def plot_graph(G, filename='prova.png', values=None, colorbar_obj=None):
    
    func_types_dic = {
                'spring' : nx.spring_layout,
                'random' : nx.random_layout,
                'shell' : nx.shell_layout,
                'spectral' : nx.spectral_layout,
                'viz' : graphviz_layout
                }

    print(G.edges())
    print(G.nodes())

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    color_nodes = [values.get(node, 0.2) for node in G.nodes()] 

    pos = func_types_dic[params.plot_type_str](G)

    nodes = nx.draw_networkx_nodes(G, pos, node_color=color_nodes, \
            alpha=.6)#, cmap=plt.get_cmap('brg'))

    nx.draw_networkx_edges(G, pos, width=2.)
    nx.draw_networkx_labels(G, pos, font_color='k', font_weight='10')
    plt.title("|V| = %d, |E| = %d"%(len(G.nodes()), len(G.edges())))
    colorbar_obj.set_array(color_nodes)
    plt.colorbar(colorbar_obj)
    plt.axis('off')
    fig.savefig(filename, format='png')
    plt.close()


def get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def plot_hist(hist, bins, fname='hist.png', threshold=None, tit_str=''):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) * .5
    plt.bar(center, hist, align='center', width=width, edgecolor='none')

    if threshold:
        ax.axvline(threshold, color='k', linestyle='--', lw=2)
    
    plt.title(tit_str)
    fig.savefig(fname, format='png')
    plt.close()


def plot_clustering(obj, data, filename, axis_str=('', ''), tit_str_add='', anot=None):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    colors = get_colors(len(obj.c_to_ind))

    for i,c in enumerate(obj.c_to_ind):
        plt.plot(*zip(*data[obj.c_to_ind[c]]), marker='o', color=colors[i], ls='', ms=4., mew=1.0, alpha=0.8, mec='none')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #plt.grid(True)
    #ax.set_axis_bgcolor('grey')
    plt.xlabel(axis_str[0])
    plt.ylabel(axis_str[1])
    fig.savefig(filename, format='png')

    plt.close()



