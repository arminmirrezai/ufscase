from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import os


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
    return pd.concat(dfs)


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


def getPath(file_name, folder_name):
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

