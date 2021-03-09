from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import json
from numpy import array


def get_corona_policy(dates: pd.DatetimeIndex, country: str):
    d_country = {'NL': 'NLD', 'ES': 'ESP', 'DE': 'DEU'}
    first_day_2020 = dates[dates.year == 2020][0].day
    covid_data = pd.read_csv(Path.cwd().absolute().parents[0].as_posix() + "/Data" + '/covid-stringency-index.csv')
    if country == 'All':
        pol_new = [covid_data[first_day_2020:][covid_data.Code == country][:(7*len(dates[dates.year >= 2020]))]
                   for country in d_country.values()]
        policy_daily = pd.concat(pol_new).groupby('Date').mean()
    else:
        policy_daily = covid_data[first_day_2020:][covid_data.Code == d_country[country]]
    policy_weekly = policy_daily.groupby(np.arange(len(policy_daily))//7).mean()[:len(dates[dates.year >= 2020])]
    policy_weekly.index = dates[dates.year >= 2020]
    x = pd.DataFrame(data=np.zeros(len(dates[dates.year < 2020])), index=dates[dates.year < 2020],
                     columns=policy_weekly.columns)
    return pd.concat([x, policy_weekly])


def get_mean_dataframe(path_to_clusters: str) -> pd.DataFrame:
    """
    Get the dataframe of cluster means to put in classes
    :param path_to_clusters: path to the cluster mean csv
    :return: dataframe
    """
    df = pd.read_csv(path_to_clusters)
    start_dates = [datetime.strptime('2016-01-01', '%Y-%m-%d') + timedelta(weeks=i) for i in range(len(df.index))]
    dfs = []
    for col in df.columns:
        if 'cluster' in col:
            df_temp = pd.DataFrame(df[col])
            df_temp['keyword'] = col
            df_temp['startDate'] = start_dates
            df_temp.columns = ['interest', 'keyword', 'startDate']
            df_temp.index = start_dates
            dfs.append(df_temp)
    df_new = pd.concat(dfs)
    df_new['country'] = "All"
    return df_new


def get_cluster_means(path_to_res: str, method: str, distance: str) -> pd.DataFrame:
    df_mean = pd.read_csv(path_to_res + '/' +method+ '_7clusters.csv')
    # if method not in df_mean[df_mean.keys()[0]].values.tolist():
    #     raise ValueError(f"Method not in list {df_mean[df_mean.keys()[0]].values.tolist()}")
    # if distance not in df_mean.keys():
    #     raise ValueError(f"Distance not in list {df_mean.keys()}")
    means = eval(df_mean[distance][0])[0]
    keywords = eval(df_mean[distance][1])[0]
    start_dates = [datetime.strptime('2016-02-28', '%Y-%m-%d') + timedelta(weeks=i) for i in range(len(means[0]))]
    dfs = []
    for i, cluster in enumerate(means, 1):
        df_temp = pd.DataFrame()
        df_temp['interest'] = cluster
        df_temp['keyword'] = 'Cluster' + str(i)
        df_temp['startDate'] = start_dates
        df_temp['cluster keywords'] = str(keywords[str(i)])
        dfs.append(df_temp)
    df_new = pd.concat(dfs)
    df_new['country'] = "All"
    df_new['method'] = method
    df_new['distance'] = distance
    return df_new


def saveResult(file_name, folder_name, df=None, txt=''):
    """
    Save the results in a new or existing folder of the day in dataframe or text file
    :param txt: text to save
    :param df: data frame to save
    :param file_name: filename required
    :param folder_name: a folder name
    """
    data_path = Path.cwd().absolute().parents[0].as_posix() + "/Data"
    folder_path = data_path + "/" + folder_name
    try:
        createDir(folder_path)
        full_path = folder_path + "/" + file_name + ".txt"
    except OSError:
        full_path = data_path + "/" + file_name + ".txt"
    if df is not None:
        df.to_csv(full_path, header=None, index=None, sep=',', mode='a')
    elif txt != '':
        file = open(full_path, 'w')
        file.write(txt)
        file.close()


def isSaved(file_name, folder_name):
    """
    Check if file is saved in Data
    """
    data_path = Path.cwd().absolute().parents[0].as_posix() + "/Data"
    return os.path.exists(data_path + "/" + folder_name + "/" + file_name + ".txt")


def getPath(file_name, folder_name=''):
    """
    Give path to data
    """
    data_path = Path.cwd().absolute().parents[0].as_posix() + "/Data"
    return data_path + "/" + folder_name + "/" + file_name + ".txt"


def createDir(path):
    """
    Create directory of multiple folders
    """
    folders = []
    curr_path = path
    while not os.path.exists(curr_path):
        if curr_path == '':
            break
        curr_path, folder = os.path.split(curr_path)
        folders.append(folder)
    for i in range(len(folders) - 1, -1, -1):
        curr_path += '/' + folders[i]
        try:
            os.mkdir(curr_path)
        except OSError:
            print("Failed to create folder: " + folders[i])
