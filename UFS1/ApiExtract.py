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
    data = GSData()
    key_words = data.key_words(country)
    if extended:
        key_words = extendKws(key_words[:2], country)
    time_interval = f"{years[0]}-01-01 {years[-1]}-12-31"

    pytrend = TrendReq(hl='en-US', timeout=(10, 25))
    pytrend.build_payload(kw_list=key_words[0], cat='71', geo=country, timeframe=time_interval)
    df_time = pytrend.interest_over_time()
    saveResult(df_time, file_name=f"{country}_{time_interval}", folder_name=('Extended'if extended else 'MotherLanguage'))
    print(f"Runtime: {time.time() - start_time}")
    return df_time


def _saveResult(df, file_name, folder_name=''):
    """
    Save the results in a new or existing folder of the day
    :param df: data frame to save
    :param file_name: filename required
    :param folder_name: a folder name
    """
    data_path = Path.cwd().absolute().parents[0].as_posix() + "/Data"
    folder_name = folder_name + '___' + date.today().strftime('%d_%m_%Y')
    folder_path = data_path + "/" + folder_name
    if os.path.exists(folder_path):
        df.to_csv(folder_path + "/" + file_name + ".txt", header=None, index=None, sep=',', mode='a')
    else:
        try:
            os.mkdir(folder_path)
            df.to_csv(folder_path + "/" + file_name + ".txt", header=None, index=None, sep=',', mode='a')
        except OSError:
            print("Failed to create folder, file saved in Data folder")
            df.to_csv(data_path + "/" + file_name + ".txt", header=None, index=None, sep=',', mode='a')

