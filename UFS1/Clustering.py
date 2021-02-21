import pandas as pd
import Description
import numpy as np
from dtw import dtw  # import dtw-python
from scipy.spatial.distance import pdist, squareform
import inspect


class Clusters:

    def __init__(self, df: pd.DataFrame, train_percentage=0.8):
        self.df_train = df[df.startDate <= df.startDate.unique()[int(train_percentage*len(df.startDate.unique()))]]
        self.df_test = df[df.startDate > df.startDate.unique()[int(train_percentage*len(df.startDate.unique()))]]
        # classes to use
        self.dd = Description.Data(df)
        self.distances = self.Distances
        self.indices = self.Scores()
        self.plot = self.Plot()

    def create_distances(self, sparsity_degree='low'):
        """
        Create the distance given a degree of sparsity
        :param sparsity_degree: degree of sparsity
        """
        if sparsity_degree not in {'no', 'low', 'mid', 'high'}:
            raise ValueError("Degree not from set {'no', 'low', 'mid', 'high'}")
        bounds = {'no': (-1.0, 0), 'low': (0, 0.3), 'mid': (0.3, 0.9), 'high':(0.9, 1.0)}
        self.distances = self.Distances(self.get_sparsity_series(bounds[sparsity_degree][0], bounds[sparsity_degree][1]))

    def hierarchical(self, distance, method):
        pass

    def k_medios(self):
        distance_matrix = self.distances.manhattan()
        result = NotImplemented
        self.save_scores(method=inspect.currentframe().f_code.co_name, distance_matrix=distance_matrix, labels=result)

    def clarans(self):
        pass

    def save_scores(self, method: str, distance_matrix, labels):
        """
        Function to save the scores of the clusters made
        :param method:
        :param distance_matrix:
        :param labels:
        :return:
        """
        scores = self.Scores(distance_matrix, labels)
        scores.Baker_Hubert_gamma()

    def interest_list(self, keyword, country):
        return self.df_train[(self.df_train.keyword == keyword) & (self.df_train.country == country)]['interest'].to_list()

    def get_sparsity_series(self, lb: float, ub: float):
        """
        Get lists of interest with lb < sparsity <= ub
        :param lb: lower bound sparsity
        :param ub: upper bound sparsity
        :return: list of lists of sparsity
        """
        unique_kws = self.df_train[['keyword', 'country']].drop_duplicates()
        return [self.interest_list(prod[0], prod[1]) for prod in unique_kws.values
                if lb < self.dd.statistics.sparsity(prod[0], prod[1]) <= ub]

    class Distances:

        def __init__(self, series):
            self.ts_matrix = np.array(series)

        def manhattan(self):
            return squareform(pdist(self.ts_matrix, 'cityblock'))

        def euclidean(self):
            pass

        def chebyshev(self):
            pass

        def dynamic_time_warping(self):
            n_series = self.ts_matrix.shape[0]
            distance_matrix = np.zeros(shape=(n_series, n_series))
            for i in range(n_series-1):
                for j in range(i + 1, self.ts_matrix.shape[0]):
                    distance_matrix[i, j] = dtw(self.ts_matrix[i, :], self.ts_matrix[j, :], keep_internals=True)
                    distance_matrix[j, i] = distance_matrix[i, j]
            return distance_matrix

        def lcss(self):
            pass

        def discrete_fourier(self):
            pass

    class Scores:

        def __init__(self, distance_matrix, labels):
            self.dm = distance_matrix
            self.labels = labels

        def Baker_Hubert_gamma(self):
            pass

        def Silhoette(self):
            pass

    class Plot:
        pass



