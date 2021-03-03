# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import ApiExtract

# years = range(2016,2022)
# data = ApiExtract.extract(years,'NL')

# # df1 = ApiExtract.extract(years,'NL')
# # df2 = ApiExtract.extract(years,'DE')
# # df3 = ApiExtract.extract(years,'ES')
# # df = pd.concat([df1, df2, df3])

# import Clustering

# clst = Clustering.Clusters(data)

# # create time series according to sparsity degree
# clst.create_ts_matrix('low')

# scores = pd.DataFrame(data={ 'manhattan' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_single' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_complete' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } },
#           'euclidean' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] }, 
#                           'hierarchical_single' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_complete' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } },
#           'chebyshev' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] }, 
#                           'hierarchical_single' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_complete' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } },
#           'dtw' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] }, 
#                           'hierarchical_single' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_complete' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } }, 
#           'lcss' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] }, 
#                           'hierarchical_single' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_complete' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } }, 
#           'dft' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] }, 
#                           'hierarchical_single' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_complete' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } }, 
#           'dwt' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] }, 
#                           'hierarchical_single' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_complete' : { 'silhouette' : [], 'gap' : [] },
#                           'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } } })


# for measure in ['manhattan', 'euclidean', 'chebyshev' 'dtw', 'lcss', 'dft', 'dwt']:
#     # create distance matrix
#     clst.create_distance_matrix(measure) 
    
#     # calculate silhouette index
#     for i in range(2, 15):
#         clst.k_medoids(i)
#         scores.loc['k_medoids'][measure]['silhouette'].append(clst.indices.Silhouette())
#         clst.hierarchical('single', i)
#         scores.loc['hierarchical_single'][measure]['silhouette'].append(clst.indices.Silhouette())
#         clst.hierarchical('complete', i)
#         scores.loc['hierarchical_complete'][measure]['silhouette'].append(clst.indices.Silhouette())
#         clst.hierarchical('ward', i)
#         scores.loc['hierarchical_ward'][measure]['silhouette'].append(clst.indices.Silhouette())
        
#     for method in {'k_medoids', 'hierarchical_single', 'hierarchical_complete', 'hierarchical_ward'}:
#         d = {'k' : list(range(2, 15)), 'Silhouette' : scores.loc[method][measure]['silhouette']}
#         sildf = pd.DataFrame(d)
#         k_opt = int(sildf[['Silhouette']].idxmax()) + 2
        
#         plt.plot(sildf.k, sildf.Silhouette, linewidth=1)
#         plt.scatter(sildf[sildf.k == k_opt].k, sildf[sildf.k == k_opt].Silhouette, s=250, c='r')
#         plt.grid(True)
#         plt.xlabel('Cluster Count')
#         plt.ylabel('Silhouette score')
#         plt.title('SILHOUETTE SCORE. Method: ' + method + ', Measure: ' + measure )
#         plt.savefig('plots/silhouette/SS_method_' + method + '_measure_' + measure + '.png')
#         plt.close()
    
#     # calculate GAP statistic
#     for method in {'k_medoids', 'hierarchical_single', 'hierarchical_complete', 'hierarchical_ward'}:
#         scores.loc[method][measure]['gap'].append(clst.GAP(method, measure))

#         # collect GAP scores     
#         k = scores.loc[method][measure]['gap'][0][0]
#         gapdf = scores.loc[method][measure]['gap'][-0][1][['clusterCount', 'gap']]
#         sd = scores.loc[method][measure]['gap'][-0][1]['s_k']
    
#         plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
#         plt.errorbar(gapdf.clusterCount, gapdf.gap, yerr=np.mean(sd), fmt='-o', capthick=1, capsize=5, linewidth=1)
#         plt.grid(True)
#         plt.xlabel('Cluster Count')
#         plt.ylabel('Gap Value')
#         plt.title('GAP STATISTIC. Method: ' + method + ', Measure: ' + measure )
#         plt.savefig('plots/gapstat/GS_method_' + method + '_measure_' + measure + '.png')
#         plt.close()
    
 
# clusters = pd.DataFrame(data={ 'manhattan' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_single' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_complete' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_ward' : { 'means' : [], 'keywords' : [] } },
#           'euclidean' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_single' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_complete' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_ward' : { 'means' : [], 'keywords' : [] } },
#           'chebyshev' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_single' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_complete' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_ward' : { 'means' : [], 'keywords' : [] } },
#           'dtw' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_single' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_complete' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_ward' : { 'means' : [], 'keywords' : [] } }, 
#           'lcss' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_single' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_complete' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_ward' : { 'means' : [], 'keywords' : [] } }, 
#           'dft' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_single' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_complete' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_ward' : { 'means' : [], 'keywords' : [] } }, 
#           'dwt' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_single' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_complete' : { 'means' : [], 'keywords' : [] },
#                           'hierarchical_ward' : { 'means' : [], 'keywords' : [] } } })
    

# for measure in {'manhattan', 'euclidean', 'chebyshev', 'dtw', 'lcss', 'dft', 'dwt'}:
#     # create distance matrix
#     clst.create_distance_matrix(measure) 
    
#     # k medoids clusters
#     clst.k_medoids(scores.loc['k_medoids'][measure]['gap'][0][0])
#     cluster_means = clst.plot.plot()
#     clusters.loc['k_medoids'][measure]['means'].append(cluster_means)
#     cluster_keywords = clst.get_cluster_keywords()
#     clusters.loc['k_medoids'][measure]['keywords'].append(cluster_keywords)
    
#     # hierarchical single clusters
#     clst.hierarchical('single', scores.loc['hierarchical_single'][measure]['gap'][0][0])
#     cluster_means = clst.plot.plot()
#     clusters.loc['hierarchical_single'][measure]['means'].append(cluster_means)
#     cluster_keywords = clst.get_cluster_keywords()
#     clusters.loc['hierarchical_single'][measure]['keywords'].append(cluster_keywords)
    
#     # hierarchical complete clusters
#     clst.hierarchical('complete', scores.loc['hierarchical_complete'][measure]['gap'][0][0])
#     cluster_means = clst.plot.plot()
#     clusters.loc['hierarchical_complete'][measure]['means'].append(cluster_means)
#     cluster_keywords = clst.get_cluster_keywords()
#     clusters.loc['hierarchical_complete'][measure]['keywords'].append(cluster_keywords)
    
#     # hierarchical ward clusters
#     clst.hierarchical('ward', scores.loc['hierarchical_ward'][measure]['gap'][0][0])
#     cluster_means = clst.plot.plot()
#     clusters.loc['hierarchical_ward'][measure]['means'].append(cluster_means)
#     cluster_keywords = clst.get_cluster_keywords()
#     clusters.loc['hierarchical_ward'][measure]['keywords'].append(cluster_keywords)
    

import matplotlib.pyplot as plt
import pandas as pd
import ApiExtract

years = range(2016,2022)
# data = ApiExtract.extract(years,'NL')

df1 = ApiExtract.extract(years,'NL')
df2 = ApiExtract.extract(years,'DE')
df3 = ApiExtract.extract(years,'ES')
data = pd.concat([df1, df2, df3])

import Clustering

clst = Clustering.Clusters(data)

# create time series according to sparsity degree
clst.create_ts_matrix('low')

scores = pd.DataFrame(data={
          'euclidean' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] }, 
                          'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } },
          'dtw' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] }, 
                          'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } }, 
          'lcss' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] }, 
                          'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } }, 
          'dft' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] }, 
                          'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } }, 
          'dwt' : { 'k_medoids' : { 'silhouette' : [], 'gap' : [] }, 
                          'hierarchical_ward' : { 'silhouette' : [], 'gap' : [] } } })


for measure in ['euclidean', 'lcss', 'dft', 'dwt', 'dtw']:
    # create distance matrix
    clst.create_distance_matrix(measure) 
    
    # calculate silhouette index
    for i in range(2, 15):
        clst.k_medoids(i)
        scores.loc['k_medoids'][measure]['silhouette'].append(clst.indices.Silhouette())
        clst.hierarchical('ward', i)
        scores.loc['hierarchical_ward'][measure]['silhouette'].append(clst.indices.Silhouette())
        
    for method in ['k_medoids', 'hierarchical_ward']:
        d = {'k' : list(range(2, 15)), 'Silhouette' : scores.loc[method][measure]['silhouette']}
        sildf = pd.DataFrame(d)
        k_opt = int(sildf[['Silhouette']].idxmax()) + 2
        
        plt.plot(sildf.k, sildf.Silhouette, linewidth=1)
        plt.scatter(sildf[sildf.k == k_opt].k, sildf[sildf.k == k_opt].Silhouette, s=250, c='r')
        plt.grid(True)
        plt.xlabel('Cluster Count')
        plt.ylabel('Silhouette score')
        plt.title('SILHOUETTE SCORE. Method: ' + method + ', Measure: ' + measure )
        plt.savefig('plots/silhouette/SS_method_' + method + '_measure_' + measure + '.png')
        plt.close()
    
    # calculate GAP statistic
    for method in ['k_medoids', 'hierarchical_ward']:
        scores.loc[method][measure]['gap'].append(clst.GAP(method, measure))

        # collect GAP scores     
        k = scores.loc[method][measure]['gap'][0][0]
        gapdf = scores.loc[method][measure]['gap'][-0][1][['clusterCount', 'gap']]
        sd = scores.loc[method][measure]['gap'][-0][1]['s_k']
    
        plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
        plt.errorbar(gapdf.clusterCount, gapdf.gap, yerr=sd, fmt='-o', capthick=1, capsize=5, linewidth=1)
        plt.grid(True)
        plt.xlabel('Cluster Count')
        plt.ylabel('Gap Value')
        plt.title('GAP STATISTIC. Method: ' + method + ', Measure: ' + measure )
        plt.savefig('plots/gapstat/GS_method_' + method + '_measure_' + measure + '.png')
        plt.close()
        
 
clusters = pd.DataFrame(data={
          'euclidean' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
                          'hierarchical_ward' : { 'means' : [], 'keywords' : [] } },
          'dtw' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
                          'hierarchical_ward' : { 'means' : [], 'keywords' : [] } }, 
          'lcss' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
                          'hierarchical_ward' : { 'means' : [], 'keywords' : [] } }, 
          'dft' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
                          'hierarchical_ward' : { 'means' : [], 'keywords' : [] } }, 
          'dwt' : { 'k_medoids' : { 'means' : [], 'keywords' : [] },
                          'hierarchical_ward' : { 'means' : [], 'keywords' : [] } } })
    

for measure in {'euclidean', 'dtw', 'lcss', 'dft', 'dwt'}:
    # create distance matrix
    clst.create_distance_matrix(measure) 
    
    # k medoids clusters
    # clst.k_medoids(scores.loc['k_medoids'][measure]['gap'][0][0])
    clst.k_medoids(7)
    cluster_means = clst.plot.plot()
    clusters.loc['k_medoids'][measure]['means'].append(cluster_means)
    cluster_keywords = clst.get_cluster_keywords()
    clusters.loc['k_medoids'][measure]['keywords'].append(cluster_keywords)
    
    # hierarchical ward clusters
    # clst.hierarchical('ward', scores.loc['hierarchical_ward'][measure]['gap'][0][0])
    clst.hierarchical('ward', 7)
    cluster_means = clst.plot.plot()
    clusters.loc['hierarchical_ward'][measure]['means'].append(cluster_means)
    cluster_keywords = clst.get_cluster_keywords()
    clusters.loc['hierarchical_ward'][measure]['keywords'].append(cluster_keywords)
    
