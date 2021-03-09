from random import seed, random
import pandas as pd
import Description
import numpy as np
from dtw import dtw
from scipy.spatial.distance import pdist, squareform
import inspect
from sklearn.metrics import silhouette_score
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.clarans import clarans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from utils import lcss_dist, hierarchical_clustering
import pywt
from statsmodels.tsa.arima_process import arma_generate_sample   

class Clusters:

    def __init__(self, df: pd.DataFrame, train_percentage=0.87):
        self.df_train = df[df.startDate <= df.startDate.unique()[int(train_percentage*len(df.startDate.unique()))]]
        self.df_test = df[df.startDate > df.startDate.unique()[int(train_percentage*len(df.startDate.unique()))]]
        self.ts_matrix = df
        self.keywords = []
        # classes to use
        self.dd = Description.Data(df)
        self.distances = self.Distances
        self.indices = self.Scores
        self.plot = self.Plot

    def create_ts_matrix(self, sparsity_degree):
        """
        Create the time series matrix given a degree of sparsity
        :param sparsity_degree: degree of sparsity
        """
        if sparsity_degree not in {'no', 'low', 'mid', 'high'}:
            raise ValueError("Degree not from set {'no', 'low', 'mid', 'high'}")
        bounds = {'no': (-1.0, 0), 'low': (-0.1, 0.3), 'mid': (0.3, 0.9), 'high':(0.9, 1.0)}
        self.distances = self.Distances(self.get_sparsity_series(bounds[sparsity_degree][0], bounds[sparsity_degree][1])[0])
        self.ts_matrix, self.keywords = self.get_sparsity_series(bounds[sparsity_degree][0], bounds[sparsity_degree][1])
        
    def get_sparsity_series(self, lb: float, ub: float):
        """
        Get lists of interest with lb < sparsity <= ub
        :param lb: lower bound sparsity
        :param ub: upper bound sparsity
        :return: list of lists of sparsity
        """
        unique_kws = self.df_train[['keyword', 'country']].drop_duplicates()
        return [self.interest_list(prod[0], prod[1]) for prod in unique_kws.values
                if lb < self.dd.statistics.sparsity(prod[0], prod[1]) <= ub], [prod for prod in unique_kws.values
                if lb < self.dd.statistics.sparsity(prod[0], prod[1]) <= ub]
    
    def save_scores(self, method: str, labels):
        """
        Function to save the scores of the clusters made
        :param method:
        :param labels:
        :return:
        """
        self.indices = self.Scores(self.ts_matrix, self.distances.dm, labels)

    def interest_list(self, keyword, country):
        return self.df_train[(self.df_train.keyword == keyword) & (self.df_train.country == country)]['interest'].to_list()
    
    def get_cluster_keywords(self):
        """
        Function to obtain the keywords of the clusters made
        :param method:
        :param labels:
        :return:
        """
        labels = self.plot.labels
        num_clusters = self.plot.num_clusters
        cluster_keywords = {}
        for i in range(num_clusters):
            cluster_keywords[str(i+1)] = []
            for j in range(len(labels)):
                if labels[j] == i + 1:
                    cluster_keywords[str((i+1))].append(self.keywords[j])
        
        return cluster_keywords
    
    def create_distance_matrix(self, measure):
        """
        Create the distance matrix given a distance measure
        :param measure: distance measure specification
        """
        if measure not in {'manhattan', 'euclidean', 'chebyshev', 'dtw', 'lcss', 'dft', 'dwt'}:
            raise ValueError("measure not from set 'manhattan', 'euclidean', 'chebyshev', 'dtw', 'lcss', 'dft', 'dwt'")
    
        if measure == 'manhattan':
            self.distances.manhattan()
            self.plot.measure = 'manhattan'
        elif measure == 'euclidean':
            self.distances.euclidean()
            self.plot.measure = 'euclidean'
        elif measure == 'chebyshev':
            self.distances.chebyshev()
            self.plot.measure = 'chebyshev'
        elif measure == 'dtw':
            self.distances.dynamic_time_warping()
            self.plot.measure = 'dtw'
        elif measure == 'lcss':
            self.distances.lcss()
            self.plot.measure = 'lcss'
        elif measure == 'dft':
            self.distances.discrete_fourier()
            self.plot.measure = 'dft'
        elif measure == 'dwt':
            self.distances.discrete_wavelet()
            self.plot.measure = 'dwt'
            
    def hierarchical(self, method, num_clusters):
        """
        Hierarchical clustering method
        :param method: linkage criterion
        :param num_clusters: desired number of clusters
        """
        if method not in {'single', 'complete', 'average', 'ward'}:
            raise ValueError("method not from set {'single', 'complete', 'average', 'ward'}")
        
        linkage_matrix = hierarchical_clustering(self.distances.dm, method)
        cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        
        # dist = 0
        # for i in range(len(cluster_labels)):
        #     for j in range(len(cluster_labels)):
        #         if cluster_labels[i] == cluster_labels[j]:
        #             dist += self.distances.dm[i, j]
        # W_k = 1/(2*num_clusters) * dist
        
        dist_clust = []
        for i in range(num_clusters):
            cluster = [x for x, e in enumerate(cluster_labels) if e == i+1]
            dist = 0
            for j in range(len(cluster)):
                for k in range(len(cluster)):   
                    if cluster[j] != cluster[k]:
                        dist += self.distances.dm[cluster[j], cluster[k]]
            
            dist = 1/(2*len(cluster)) * dist
            dist_clust.append(dist)
            
        W_k = sum(dist_clust)
        
        measure = self.plot.measure
        self.save_scores(method=inspect.currentframe().f_code.co_name, labels=cluster_labels)
        self.plot = self.Plot(series = self.ts_matrix, labels=cluster_labels, num_clusters=num_clusters)
        
        if method == 'single':
            self.plot.method = 'hierarchical_single'
        elif method == 'complete':
            self.plot.method = 'hierarchical_complete'
        elif method == 'ward':
            self.plot.method = 'hierarchical_ward'
        self.plot.measure = measure
        
        return W_k

    def k_medoids(self, num_clusters):
        initial_medoids =  np.random.uniform(0,len(self.distances.dm), num_clusters).astype(int).tolist()
        kmedoids_instance = kmedoids(self.distances.dm, initial_medoids, data_type='distance_matrix', itermax=100000)
        #kmedoids_instance = kmedoids(self.ts_matrix, initial_medoids, itermax=100000)
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()
        # medoids = kmedoids_instance.get_medoids()
        cluster_labels = np.zeros(len(self.distances.dm))

        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                cluster_labels[clusters[i][j]] = i+1

        # dist = 0
        # for i in range(len(cluster_labels)):
        #     for j in range(len(cluster_labels)):
        #         if cluster_labels[i] == cluster_labels[j]:
        #             dist += self.distances.dm[i, j]
        
        dist_clust = []
        for i in range(num_clusters):
            dist = 0
            for j in range(len(clusters[i])):
                for k in range(len(clusters[i])):   
                    if clusters[i][j] != clusters[i][k]:
                        dist += self.distances.dm[clusters[i][j], clusters[i][k]]
            
            dist = 1/(2*len(clusters[i])) * dist
            dist_clust.append(dist)
            
        W_k = sum(dist_clust)
            

        measure = self.plot.measure
        self.save_scores(method=inspect.currentframe().f_code.co_name, labels=cluster_labels)
        self.plot = self.Plot(series = self.ts_matrix, labels=cluster_labels, num_clusters=num_clusters)
        self.plot.method = 'k_medoids'
        self.plot.measure = measure
        
        
        return W_k

    def clarans(self, num_clusters):
        clarans_instance = clarans(data=self.ts_matrix, number_clusters=num_clusters, numlocal=6, maxneighbor=6)
        clarans_instance.process()
        clusters = clarans_instance.get_clusters()
        cluster_labels = np.zeros(len(self.distances.dm))
        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                cluster_labels[clusters[i][j]] = i+1
             
        self.save_scores(method=inspect.currentframe().f_code.co_name, labels=cluster_labels)
        self.plot = self.Plot(series = self.ts_matrix, labels=cluster_labels, num_clusters=num_clusters)
        
    def GAP(self, method, measure, nrefs=5, maxClusters=10):
        """
        Calculates optimal number of clusters using Gap Statistic from Tibshirani, Walther, Hastie
        :param nrefs: number of sample reference datasets to create
        :param maxClusters: Maximum number of clusters to test for
        Returns: (k_opt, resultsdf)
        """
        if method not in {'k_medoids', 'hierarchical_single', 'hierarchical_complete', 'hierarchical_ward'}:
            raise ValueError("method not from set {'k_medoids', 'hierarchical_complete', 'hierarchical_ward'}")
            
        if measure not in {'manhattan', 'euclidean', 'chebyshev', 'dtw', 'lcss', 'dft', 'dwt'}:
            raise ValueError("measure not from set 'manhattan', 'euclidean', 'chebyshev', 'dtw', 'lcss', 'dft', 'dwt'")
            
        self.create_distance_matrix(measure)
        dm = self.distances.dm
            
        seed(123)
        gaps = np.zeros((len(range(1, maxClusters)),))
        s_ks = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[], 's_k':[]})
        
        for gap_index, k in enumerate(range(1, maxClusters)):
    
            # Holder for reference dispersion results
            refDisps = np.zeros(nrefs)
    
            # For n references, generate random sample and perform clutsering getting resulting dispersion of each loop
            for i in range(nrefs):
                
                # Create new random reference set
                # ts_matrix_random = []
                # for j in range(len(self.distances.ts_matrix)):
                #     random_walk = [np.random.uniform(0,90)]
                #     for m in range(1, len(self.distances.ts_matrix[0])):
                #         movement = -1 if random() < 0.5 else 1
                #         value = random_walk[m-1] + movement
                #         random_walk.append(value)
                #     random_walk_norm = [100*(float(l)-min(random_walk))/(max(random_walk)-min(random_walk)) for l in random_walk]
                #     ts_matrix_random.append(random_walk_norm)
                   
                ts_matrix_random = []
                for j in range(len(self.distances.ts_matrix)):
                    arparams = np.array([.75, -.25])
                    maparams = np.array([.65, .35])
                    ar = np.r_[1, -arparams] # add zero-lag and negate
                    ma = np.r_[1, maparams] # add zero-lag
                    y = arma_generate_sample(ar, ma, len(self.distances.ts_matrix[0]))
                    y_norm =  [100*(float(l)-min(y))/(max(y)-min(y)) for l in y]
                    ts_matrix_random.append(y_norm)
                
                self.distances.ts_matrix =  np.array(ts_matrix_random)
                self.create_distance_matrix(measure)
                
                # Fit to it
                if method == 'k_medoids':
                    refDisp = self.k_medoids(k)
                elif method == 'hierarchical_single':
                    refDisp = self.hierarchical('single', k)
                elif method == 'hierarchical_complete':
                    refDisp = self.hierarchical('complete', k)
                elif method == 'hierarchical_ward':
                    refDisp = self.hierarchical('ward', k)
                refDisps[i] = refDisp
    
            # Fit cluster to original data and create dispersion
            self.distances.ts_matrix = np.array(self.ts_matrix)
            self.distances.dm = dm
            
            if method == 'k_medoids':
                origDisp = self.k_medoids(k)
            elif method == 'hierarchical_single':
                origDisp = self.hierarchical('single', k)
            elif method == 'hierarchical_complete':
                origDisp = self.hierarchical('complete', k)
            elif method == 'hierarchical_ward':
                origDisp = self.hierarchical('ward', k)
            
            # Calculate gap statistic
            gap = np.mean(np.log(refDisps)) - np.log(origDisp)
            sd_k = np.std(np.log(refDisps))
            s_k = sd_k * np.sqrt(1+1/nrefs)
            
            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap
            s_ks[gap_index] = s_k
            
            resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap, 's_k': s_k}, ignore_index=True)
        
        # if method == 'k_medoids':
        if np.nonzero(np.array(gaps[0:(maxClusters-2)]) >= np.array((gaps[1:(maxClusters-1)] - s_ks[1:(maxClusters-1)])))[0]:
            k_opt = np.nonzero(np.array(gaps[0:(maxClusters-2)]) >= np.array((gaps[1:(maxClusters-1)] - s_ks[1:(maxClusters-1)])))[0][0] + 1
        else:
            k_opt = gaps.argmax() + 1
            
        return (k_opt, resultsdf) 
    

    class Distances:

        def __init__(self, series):
            self.ts_matrix = np.array(series)
            self.dm = np.zeros(shape=(self.ts_matrix.shape[0], self.ts_matrix.shape[0]))

        def manhattan(self):
            self.dm = squareform(pdist(self.ts_matrix, 'cityblock'))

        def euclidean(self):
            self.dm = squareform(pdist(self.ts_matrix, 'euclidean'))

        def chebyshev(self):
            self.dm = squareform(pdist(self.ts_matrix, 'chebyshev'))

        def dynamic_time_warping(self):
            n_series = self.ts_matrix.shape[0]
            for i in range(n_series):
                for j in range(n_series):
                    x = self.ts_matrix[i, :].tolist()
                    y = self.ts_matrix[j, :].tolist()
                    if i != j:
                        dist = dtw(x, y, keep_internals=True).distance
                        self.dm[i, j] = dist
        
        def lcss(self):
            n_series = self.ts_matrix.shape[0]            
            for i in range(n_series):
                for j in range(n_series):
                    x = self.ts_matrix[i, :].tolist()
                    y = self.ts_matrix[j, :].tolist()
                    if i != j:
                        dist = lcss_dist(x, y, delta=np.inf, epsilon=0.5)
                        self.dm[i, j] = dist

        def discrete_fourier(self):
            n_series = self.ts_matrix.shape[0]            
            for i in range(n_series):
                for j in range(n_series):
                    x = self.ts_matrix[i, :].tolist()
                    y = self.ts_matrix[j, :].tolist()
                    if i != j:
                        xdft = np.fft.rfft(x)
                        ydft = np.fft.rfft(y)
                        num_freq = 50
                        dist = np.linalg.norm(np.asarray(sorted(xdft, reverse=True)[0:num_freq]) - np.asarray(sorted(ydft, reverse=True)[0:num_freq])) 
                        # dist = np.linalg.norm(np.asarray(sorted(xdft, reverse=True)) - np.asarray(sorted(ydft, reverse=True))) 
                        self.dm[i, j] = dist
        
        def discrete_wavelet(self):
            n_series = self.ts_matrix.shape[0]            
            for i in range(n_series):
                for j in range(n_series):
                    x = self.ts_matrix[i, :].tolist()
                    y = self.ts_matrix[j, :].tolist()
                    if i != j:
                        Xcoeffs = pywt.downcoef('a', x, 'sym8', level=4)                    
                        Ycoeffs = pywt.downcoef('a', y, 'sym8', level=4)   
                         
                        dist = np.linalg.norm(Xcoeffs - Ycoeffs) 
                        self.dm[i, j] = dist


    class Scores:

        def __init__(self, ts_matrix, distance_matrix, labels):
            self.ts_matrix = ts_matrix
            self.dm = distance_matrix
            self.labels = labels

        
        def Silhouette(self):
            return silhouette_score(self.dm, self.labels, metric="precomputed")


    class Plot:
        
        def __init__(self, series, labels, num_clusters):
            self.series = series
            self.labels = labels
            self.num_clusters = num_clusters
            self.measure = str()
            self.method  = str()
            
        def plot(self, save=True):
            
            cluster_means = []    
            for i in range(self.num_clusters):
                cluster = np.where(self.labels == i+1)
                mean_series = np.zeros(np.array(self.series[cluster[0][0]]).size)
                for j in range(cluster[0].size):
                    mean_series += np.array(self.series[cluster[0][j]])
                    plt.plot(self.series[cluster[0][j]], 'k')
                mean_series = mean_series / cluster[0].size
                cluster_means.append(mean_series)
                plt.plot(mean_series, 'r')
                plt.title('Method: ' + self.method + ', Measure: ' + self.measure + ', Num_clusters: ' + str(self.num_clusters) )
                if save==True:
                    plt.savefig('plots/clusters/k='+ str(self.num_clusters) +'/'+ self.method + '/' + self.measure + '/clust'+str(i+1)+'.png')
                else:
                    plt.show()
                plt.close()
                
            return cluster_means
        
        
    
                
    
        