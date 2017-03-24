import pandas as pd
import numpy as np
import explore_mapper as em
from scipy.spatial.distance import cdist, pdist
from em_help import *

try:
    import params
except ImportError:
    import params_default as params



def find_opt_threshold(hist, bin_edges, limit=3):


    infl_points = []

    deltas = [hist[i+1] - hist[i] for i in range(len(hist)-1)]
    
    infl_points = [i for i in range(1, len(hist)-1) if np.sign(deltas[i]) != np.sign(deltas[i-1])]
    infl_points = [0]+infl_points

    if len(infl_points) == 1:
        return bin_edges[np.argmin(deltas)+1]
         


    infl_deltas = [(hist[infl_points[i+1]] - hist[infl_points[i]], infl_points[i+1]) for i in range(len(infl_points)-1)][:-1]   

    thresh_ind = min(list(filter(lambda x: x[0]<0, infl_deltas)), key=lambda x: x[1])[1]

    return bin_edges[thresh_ind]
    



class SingleLinkageClustering(em.ClusteringTDA):


    def __init__(self, data):
        self.data = data
        self.k = params.CLUSTERING_BIN_NUMBER
        self.resolution = 0

        self.var_vec = [v if v > 0 else 1. for v in np.var(data, axis=0)]
        
        
        self.indices = np.arange(len(data))
        self.ind_to_c = {}
        self.c_to_ind = {}



    def run_clustering(self, lower_popul_bound=2):
 
        self.resolution, self.hist, self.bin_edges = self.compute_thresh()

        self.tad_algo()

        for c in list(self.c_to_ind.keys()):
            if len(self.c_to_ind[c]) <= lower_popul_bound:
                self.c_to_ind.pop(c, None)
        
        return self.c_to_ind


    def make_plot(self, plot_name):
        tit_str = "n_data = %d, b_bins = %d"%(len(self.data), len(self.hist))
        plot_hist(self.hist, self.bin_edges, fname=plot_name, threshold=self.resolution)


    def compute_thresh(self):

        flat_adj_matrix = pdist(self.data, metric='seuclidean', V=self.var_vec)

        hist, bin_edges = np.histogram(flat_adj_matrix, bins=self.k)

        opt_thresh = find_opt_threshold(hist, bin_edges, limit=3)

        return opt_thresh, hist, bin_edges
        


    def cdistance_norm(self, a, b):
        return cdist(a, b, metric='seuclidean',V=self.var_vec)[0]


    def merge_clusters(self, neighbor_clusters, nodes):
        
        external_nodes = []    

        for c in neighbor_clusters:
            external_nodes.extend( self.c_to_ind[c] )
            self.c_to_ind.pop(c, None)


        
        return list(set(external_nodes)|set(nodes))

    def update_cluster_membership(self, cluster_name):
        return list(zip(self.c_to_ind[cluster_name], [cluster_name]*len(self.c_to_ind[cluster_name])))


    def tad_algo(self):

        cluster_name = 0
        
        for i in self.indices:

            if i not in self.ind_to_c:
                
                dists_i = self.cdistance_norm(self.data[i:i+1], self.data)
                nodes = self.indices[ dists_i < self.resolution ]

                neighbor_clusters = set([self.ind_to_c[n] for n in nodes if n in self.ind_to_c])
                
                self.c_to_ind[cluster_name] = self.merge_clusters(neighbor_clusters, nodes)

                clus_mbrship = self.update_cluster_membership(cluster_name)
                
                self.ind_to_c.update( clus_mbrship )
                
                cluster_name += 1
        


class NNC(em.ClusteringTDA):


    def __init__(self, data):
        self.data = data
        self.k = 8

        self.var_vec = [v if v > 0 else 1. for v in np.var(data, axis=0)]
        
        self.indices = np.arange(len(data))
        self.ind_to_c = {}
        self.c_to_ind = {}



    def run_clustering(self, lower_popul_bound=0):

        self.knn_algo()


        for c in list(self.c_to_ind.keys()):
            if len(self.c_to_ind[c]) <= lower_popul_bound:
                self.c_to_ind.pop(c, None)
        
        return self.c_to_ind


    def make_plot(self, plot_name):
        pass


    def cdistance_norm(self, a, b):
        return cdist(a, b, metric='seuclidean',V=self.var_vec)[0]


    def merge_clusters(self, neighbor_clusters, nodes):
        
        external_nodes = []    

        for c in neighbor_clusters:
            external_nodes.extend( self.c_to_ind[c] )
            self.c_to_ind.pop(c, None)


        
        return list(set(external_nodes)|set(nodes))

    def update_cluster_membership(self, cluster_name):
        return list(zip(self.c_to_ind[cluster_name], [cluster_name]*len(self.c_to_ind[cluster_name])))


    def nnc_algo(self):

        cluster_name = 0

        for i in self.indices:

            if i not in self.ind_to_c:
                
                dists_i = self.cdistance_norm(self.data[i:i+1], self.data)
                

                nodes = self.indices[ np.argsort(dists_i)[:self.k] ][1:]

                neighbor_clusters = set([self.ind_to_c[n] for n in nodes if n in self.ind_to_c])
                
                self.c_to_ind[cluster_name] = self.merge_clusters(neighbor_clusters, nodes)

                clus_mbrship = self.update_cluster_membership(cluster_name)
                
                self.ind_to_c.update( clus_mbrship )
                
                cluster_name += 1



if __name__ == '__main__':


    data1 = np.random.multivariate_normal(mean=[0,0], cov=[[50,0],[0,40]], size=100)

    data2 = np.random.multivariate_normal(mean=[100,100], cov=[[30,0],[0,30]], size=100)

    data = np.array(list(data1) + list(data2))

    var = SingleLinkageClustering(data)
    var.run_slc()

    plot_TAD(var, data, 'prova.png')









