from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
import pandas as pd
import time

from pathlib import Path
import os
from datetime import date
from SearchData import GSData


def extract(years, country, extended=True):
    start_time = time.time()
    gd = GSData()
    key_words = gd.load_key_words(country, translated=extended)
    time_interval = f"{years[0]}-01-01 {years[-1]}-12-31"

    folder_name = ('Extended' if extended else 'Simple') + f"/{time_interval}/{country}/"
    pytrend = TrendReq(hl='en-US', timeout=(10, 25))
    frames = []
    missed = 0
    for i, key_word in key_words.iterrows():
        file_name = key_word[country]
        if not isSaved(file_name, folder_name):
            kw_list = [key_word[country], key_word['EN']] if extended else [key_word[country]]
            if kw_list[0] == kw_list[1]:
                kw_list = [kw_list[0]]
            try:
                pytrend.build_payload(kw_list, cat='71', geo=country, timeframe=time_interval)
                df_time = pytrend.interest_over_time()
                saveResult(df_time, file_name=file_name, folder_name=folder_name)
            except ResponseError:
                missed += 1
                print("Time out because of response error")
                time.sleep(5)
        else:
            df_time = pd.read_csv(getPath(file_name, folder_name))
        frames.append(df_time)
        print(f"Number of words {i + 1} done")
    print(f"Runtime: {time.time() - start_time} for country {country}")
    if missed == 0:
        return frames
    else:
        return extract(years, country)


def saveResult(df, file_name, folder_name):
    """
    Save the results in a new or existing folder of the day
    :param df: data frame to save
    :param file_name: filename required
    :param folder_name: a folder name
    """
    data_path = Path.cwd().absolute().parents[0].as_posix() + "/Data"
    folder_path = data_path + "/" + folder_name
    try:
        createDir(folder_path)
        df.to_csv(folder_path + "/" + file_name + ".txt", header=None, index=None, sep=',', mode='a')
    except OSError:
        df.to_csv(data_path + "/" + file_name + ".txt", header=None, index=None, sep=',', mode='a')


def isSaved(file_name, folder_name):
    data_path = Path.cwd().absolute().parents[0].as_posix() + "/Data"
    return os.path.exists(data_path + "/" + folder_name + "/" + file_name + ".txt")


def getPath(file_name, folder_name):
    data_path = Path.cwd().absolute().parents[0].as_posix() + "/Data"
    return data_path + "/" + folder_name + "/" + file_name + ".txt"


def createDir(path):
    folders = []
    curr_path = path
    while not os.path.exists(curr_path):
        if curr_path == '':
            break
        curr_path, folder = os.path.split(curr_path)
        folders.append(folder)
    for i in range(len(folders)-1, -1, -1):
        curr_path += '/' + folders[i]
        try:
            os.mkdir(curr_path)
        except OSError:
            print("Failed to create folder: " + folders[i])
