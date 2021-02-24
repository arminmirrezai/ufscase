from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
import pandas as pd
from pandas.errors import EmptyDataError
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from SearchData import GSData
from DataUtil import *

gd = GSData()


def extract(years, country, extended=True) -> pd.DataFrame:
    """
    Extract the data from google trends given the time interval and country
    :param years: number of years
    :param country: country of choice
    :param extended: True if you also want to include native language and native
    :return: array with data frames of searched data
    """
    start_time = time.time()
    key_words = gd.load_key_words(country, translated=extended)
    time_interval = f"{years[0]}-01-01 {years[-1]}-12-31"

    folder_name = ('Extended' if extended else 'Simple') + f"/{time_interval}/{country}"
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
                saveResult(df=df_time, file_name=file_name, folder_name=folder_name)
                if not df_time.empty:
                    frames.append(adjustDataframe(df_time, getPath(file_name, folder_name)))
            except ResponseError:
                missed += 1
                print("Time out because of response error")
                time.sleep(5)
            print(f"Number of words {i + 1 - missed} done")
        else:
            try:
                df_time = pd.read_csv(getPath(file_name, folder_name), header=None)
                frames.append(adjustDataframe(df_time, getPath(file_name, folder_name)))
            except EmptyDataError:
                print(f"The file for {key_word[country]} is empty")
    if missed == 0:
        print(f"Runtime: {time.time() - start_time} for country {country} completed")
        return pd.concat(frames)
    else:
        print(f"Runtime: {time.time() - start_time} for country {country}, still missed {missed} "
              f"words and has to run again")
        return extract(years, country)


def adjustDataframe(df: pd.DataFrame, path: str):
    """
    Adjust dataframe to also get information
    :param df: datafram of just interest
    :param path: path to dataframe
    :return: new extended dataframe
    """
    dir_names = path.split(sep='/')
    keyword = dir_names[-1][:-4]
    country = dir_names[-2]
    time_interval = dir_names[-3].split()
    t0 = datetime.strptime(time_interval[0], '%Y-%m-%d')
    T = datetime.strptime(time_interval[1], '%Y-%m-%d')
    if (T - t0).days > (4 * 365 + 366):
        start_dates = [t0 + relativedelta(months=+i) for i in range(len(df.index))]
        end_dates = [start_dates[i] + relativedelta(months=+1) - timedelta(days=1) for i in range(len(df.index))]
    else:
        start_dates = [t0 + timedelta(weeks=i) for i in range(len(df.index))]
        end_dates = [start_dates[i] + timedelta(days=6) for i in range(len(df.index))]

    # Merge country language and english into one
    if len(df.columns) == 3:
        df_new = pd.DataFrame(df.iloc[:, 0] + df.iloc[:, 1], columns=['interest'])
        df_new['interest'] = df_new['interest'].div(df_new['interest'].max() / 100)
    else:
        df_new = df.drop(df.columns[1], axis=1)
        df_new.columns = ['interest']
    df_new['keyword'] = keyword
    df_new['category'] = gd.getCategory(keyword, country)
    df_new['startDate'] = pd.Series(start_dates)
    df_new['endDate'] = pd.Series(end_dates)
    df_new['country'] = country
    return df_new
