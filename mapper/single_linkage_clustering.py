import pandas as pd
import numpy as np
import explore_mapper as em
from scipy.spatial.distance import cdist, pdist
from em_help import *

try:
    import params
except ImportError:
    import params_default as params

####################
# 1. Make curve fit (avoids noise) f to dist
# 2. Find local minimums of f
# 3. Define heuristics to choose opt. minima
####################

def find_opt_threshold(hist, bin_edges, limit=3):
    
    sort_ind = np.lexsort((list(range(len(hist))), hist))


    for i in sort_ind:
        left = i
        right = i
        counter = 0
        while left != 0 and right != len(sort_ind)-1:
            left -= 1
            right += 1
            if hist[i] < hist[left] and hist[i] < hist[right]:
                counter += 1
            if counter == limit:
                return bin_edges[i]

    return bin_edges[-1]
####################




class SingleLinkageClustering(em.ClusteringTDA):


    def __init__(self, data):
        self.data = data
        self.k = params.CLUSTERING_BIN_NUMBER
        self.resolution = 0

        self.var_vec = [v if v > 0 else 1. for v in np.var(data, axis=0)]
        
        
        self.indices = np.arange(len(data))
        self.ind_to_c = {}
        self.c_to_ind = {}



    def run_clustering(self):

        self.resolution, self.hist, self.bin_edges = self.compute_thresh()

        self.tad_algo()
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
        #print set(self.ind_to_c.values()), self.c_to_ind.keys()
        external_nodes = []    

        for c in neighbor_clusters:
            external_nodes.extend( self.c_to_ind[c] )
            self.c_to_ind.pop(c, None)


        #print external_nodes, nodes
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
        


if __name__ == '__main__':


    data1 = np.random.multivariate_normal(mean=[0,0], cov=[[50,0],[0,40]], size=100)

    data2 = np.random.multivariate_normal(mean=[100,100], cov=[[30,0],[0,30]], size=100)

    data = np.array(list(data1) + list(data2))

    var = SingleLinkageClustering(data)
    var.run_slc()

    plot_TAD(var, data, 'prova.png')









