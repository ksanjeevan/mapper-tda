import pandas as pd
import numpy as np
import abc

try:
    import params
except ImportError:
    import params_default as params
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from em_help import *


class ExploreMapper:
    """
    Implementation of the Mapper class, TDA techniques.
    """

    def __init__(self, data, Clustering, FilterFunction):

        
        self.N = params.N
        self.p = params.p

        self.data = data
        self.indices = np.arange(len(data))

        
        self.filter_class = FilterFunction
        self.cluster_class = Clustering

        
        self._implem_check()

        self.run_mapper()



    def _implem_check(self):

        if not issubclass(self.cluster_class, ClusteringTDA):
            raise TypeError('Clustering class must implement \'ClusteringTDA\' abstract class.')

        if not issubclass(self.filter_class, FilterFuctionTDA):
            raise TypeError('Filter Function class must implement \'FilterFuctionTDA\' abstract class.')



    def _apply_filter_function(self):
        fm = []
        filter_obj = self.filter_class(self.data)

        for i in self.indices:
            fm.append(filter_obj.filter_func(self.data[i:i+1], self.data))

        return pd.Series(fm, index=self.indices).sort_values()


    def _apply_clustering(self):
        binned_dict, bins = self._bin_data()

        partial_clusters = {}
        counter = 0
        node_colors = {}

        clusters = []
        
        for i, k in enumerate(bins):

            keys = list(binned_dict[k].index)

            local_to_global = dict(zip(list(range(len(self.data))), keys))

            cluster_obj = self.cluster_class(self.data[keys])

            clusters.append( cluster_obj )

            c_to_ind = cluster_obj.run_clustering()

            global_cluster_names = {}
            for c in c_to_ind:
                global_cluster_names[counter] = [local_to_global[local_index] for local_index in c_to_ind[c]]
                node_colors[counter] = np.mean(binned_dict[k])
                counter += 1

            partial_clusters[i] = global_cluster_names


        if params.CLUSTERING_PLOT_BOOL:
            for i, c in enumerate(clusters):
                c.make_plot(plot_name=params.PLOT_PATH+'hist_%s.png'%(i))


        return partial_clusters



    def _build_graph(self, partial_clusters):

        node_colors, colorbar_obj = self._get_colors(partial_clusters)

        self._plot_3d_color(node_colors, partial_clusters)
        
        self._plot_3d_clusters(node_colors, partial_clusters)

        G = nx.Graph()

        for k in range(len(partial_clusters)-1):
            for c in partial_clusters[k]:
                G.add_node(c)

        for k in range(len(partial_clusters)-1):
            for c1 in partial_clusters[k]:
                for c2 in partial_clusters[k+1]:
                    if set(partial_clusters[k][c1]).intersection(partial_clusters[k+1][c2]):
                        G.add_edge(c1, c2)


        self._plot_3d_graph(node_colors, partial_clusters, G.edges())
        plot_graph(G, values=node_colors, colorbar_obj=colorbar_obj, filename='graph.png')

        

    def _get_colors(self, dic, cmap_str='brg'):

        cNorm  = colors.Normalize(vmin=min(self.filtered_values), vmax=max(self.filtered_values))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap_str))

        node_color_dic = {}
        for _, clusters in dic.items():
            for node, indices in clusters.items():
                mitjana = np.mean(self.filtered_values[indices])
                node_color_dic[node] = scalarMap.to_rgba(mitjana)

        return node_color_dic, scalarMap




    def _bin_data(self):
        """
         Bin filter function array into N bins with percent overlap p
         Return filter function bin membership and bin edges
        """

        finish = self.filtered_values.iloc[-1]
        start = self.filtered_values.iloc[0]

        # Size of bins, bin overlap size, bins
        bin_len = (finish-start)/self.N
        bin_over = self.p*bin_len
        bins = [(start + (bin_len-bin_over)*i, start + bin_len*(i+1)) for i in range(self.N)]

        binned_dict = {}
        for edge in bins:
            bool_corr = self.filtered_values.apply(lambda x: True if x>=edge[0] and x<=edge[1] else False)
            binned_dict[edge] = self.filtered_values[bool_corr]


        return binned_dict, bins


    def _plot_3d_clusters(self, node_colors, partial_clusters):

        for i, clusters in partial_clusters.items():
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            #ax.scatter(*zip(*self.data), marker='o', s=10, c='k', alpha=0.2, lw=0)
            plt.plot(*zip(*self.data), marker='o', color='k', ls='', ms=3., mew=1.0, alpha=0.2, mec='none')
            
            colors = get_colors(len(clusters))
            j = 0
            for node, indices in clusters.items():
                #color = node_colors[node]
                #ax.scatter(*zip(*self.data[indices]), marker='o', s=20, c=colors[j], alpha=1., lw=0, label="Node %d"%(node))   
                ax.plot(*zip(*self.data[indices]), marker='o', color=colors[j], ls='', ms=4., mew=1.0, \
                    alpha=0.5, mec='none', label="Node %d"%(node))
                j += 1

            #ax.legend(loc='upper right', scatterpoints=1)
            ax.legend(loc='upper right', numpoints=1, scatterpoints=1)

            ax.view_init(*params.ANGLE)
            fig.savefig(params.PLOT_PATH + "%s.png"%(i), format='png')
            plt.close()


    def _plot_3d_color(self, node_colors, partial_clusters, fname='3d_coloring.png'):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for _, clusters in partial_clusters.items():
            for node, indices in clusters.items():
                color = node_colors[node]
                ax.scatter(*zip(*self.data[indices]), s=20, c=color, lw=0, alpha=.4)

        ax.view_init(*params.ANGLE)
        fig.savefig(fname, format='png')
        plt.close()


    def _plot_3d_graph(self, node_colors, partial_clusters, edges, fname='3d_graph.png'):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        c_to_centroid = {}
        for _, clusters in partial_clusters.items():
            for node, indices in clusters.items():
                color = node_colors[node]
                ax.scatter(*np.mean(self.data[indices], axis=0), s=200, c=color, lw=0, alpha=.4)
                c_to_centroid[node] = np.mean(self.data[indices], axis=0)

        for e in edges:
            e1, e2 = e
            x = c_to_centroid[e1]
            y = c_to_centroid[e2]
            ax.plot(*zip(x,y), ms=0, ls='-', lw=1., color='k')

        ax.view_init(*params.ANGLE)
        fig.savefig(fname, format='png')
        plt.close()


    def run_mapper(self):

        print("Applying Filter Function...")
        print("--------------------------------")

        # Store filter function results array
        self.filtered_values = self._apply_filter_function()

        print("Start Partial Clustering...")
        print("--------------------------------")

        # Apply clustering to each of the bins
        partial_clusters = self._apply_clustering()

        print("Building Graph...")
        print("--------------------------------")

        # Build edges between clusters of different bins if they share points
        self._build_graph(partial_clusters)


import abc

class ClusteringTDA(abc.ABC):

    """
    Abstract Clustering class to be implemented for Mapper
    """

    def __init__(self, data):
        pass

    @abc.abstractmethod
    def run_clustering(self):
        pass

    @abc.abstractmethod
    def make_plot(self, plot_name):
        pass



class FilterFuctionTDA(abc.ABC):

    """
    Abstract Filter Function class to be implemented for Mapper
    """

    def __init__(self, data):
        pass

    @abc.abstractmethod
    def filter_func(self, *args):
        pass





if __name__ == '__main__':
    pass















