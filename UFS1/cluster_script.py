import matplotlib.pyplot as plt
import pandas as pd
import ApiExtract

# initialize year range
years = range(2016,2022)

# extract data
df1 = ApiExtract.extract(years,'NL')
df2 = ApiExtract.extract(years,'DE')
df3 = ApiExtract.extract(years,'ES')
# df4 = ApiExtract.extract(years,'GB')
# df5 = ApiExtract.extract(years,'SE')
# df6 = ApiExtract.extract(years,'FR')
# df7 = ApiExtract.extract(years,'IT')
# df8 = ApiExtract.extract(years,'PT')
data = pd.concat([df1, df2, df3])

# create clustering class
import Clustering
clst = Clustering.Clusters(data)

# create time series according to sparsity degree
clst.create_ts_matrix('low')

# create dataframe of scores 
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

# for each distance measure and clustering method
for measure in ['euclidean', 'lcss', 'dft', 'dwt', 'dtw']:
    # create distance matrix
    clst.create_distance_matrix(measure) 
    
    scores.loc['k_medoids'][measure]['silhouette'].append(0)
    scores.loc['hierarchical_ward'][measure]['silhouette'].append(0)
    
    # calculate silhouette index
    for i in range(2, 10):
        clst.k_medoids(i)
        scores.loc['k_medoids'][measure]['silhouette'].append(clst.indices.Silhouette())
        clst.hierarchical('ward', i)
        scores.loc['hierarchical_ward'][measure]['silhouette'].append(clst.indices.Silhouette())
        
    # plot silhouette scores
    for method in ['k_medoids', 'hierarchical_ward']:
        d = {'k' : list(range(1, 10)), 'Silhouette' : scores.loc[method][measure]['silhouette']}
        sildf = pd.DataFrame(d)
        k_opt = int(sildf[['Silhouette']].idxmax()) + 1
        
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

        # collect GAP statistics    
        k = scores.loc[method][measure]['gap'][0][0]
        gapdf = scores.loc[method][measure]['gap'][-0][1][['clusterCount', 'gap']]
        sd = scores.loc[method][measure]['gap'][-0][1]['s_k']
        
        # plot GAP statistics
        plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
        plt.errorbar(gapdf.clusterCount, gapdf.gap, yerr=sd, fmt='-o', capthick=1, capsize=5, linewidth=1)
        plt.grid(True)
        plt.xlabel('Cluster Count')
        plt.ylabel('Gap Value')
        plt.title('GAP STATISTIC. Method: ' + method + ', Measure: ' + measure )
        plt.savefig('plots/gapstat/GS_method_' + method + '_measure_' + measure + '.png')
        plt.close()
        

# calculate clusters for k = [7,10]
for i in range(7, 10):
    # initialize dataframe for k_medoids
    clusters_kmed = pd.DataFrame(data={
          'euclidean' : { 'means' : [], 'keywords' : [] },
          'dtw' : { 'means' : [], 'keywords' : [] }, 
          'lcss' : { 'means' : [], 'keywords' : [] },
          'dft' : { 'means' : [], 'keywords' : [] },
          'dwt' : { 'means' : [], 'keywords' : [] }})
        
    # initialize dataframe for hierarchical ward
    clusters_ward = pd.DataFrame(data={
              'euclidean' : { 'means' : [], 'keywords' : [] },
              'dtw' : { 'means' : [], 'keywords' : [] }, 
              'lcss' : { 'means' : [], 'keywords' : [] },
              'dft' : { 'means' : [], 'keywords' : [] },
              'dwt' : { 'means' : [], 'keywords' : [] }})
    
    # calculate means, extract cluster keywords and save plots
    for measure in {'euclidean', 'dtw', 'lcss', 'dft', 'dwt'}:
        # create distance matrix
        clst.create_distance_matrix(measure) 
        
        # k medoids clusters
        clst.k_medoids(i)
        cluster_means = clst.plot.plot()
        clusters_kmed.loc['means'][measure].append(cluster_means)
        cluster_keywords = clst.get_cluster_keywords()
        clusters_kmed.loc['keywords'][measure].append(cluster_keywords)
        
        # hierarchical ward clusters
        clst.hierarchical('ward', i)
        cluster_means = clst.plot.plot()
        clusters_ward.loc['means'][measure].append(cluster_means)
        cluster_keywords = clst.get_cluster_keywords()
        clusters_ward.loc['keywords'][measure].append(cluster_keywords)
        
    clusters_kmed.to_csv('plots/clusters/k='+ str(i) +'/clusters_kmed_'+str(i)+'.csv') 
    clusters_ward.to_csv('plots/clusters/k='+ str(i) +'/clusters_ward_'+str(i)+'.csv') 
    
    


