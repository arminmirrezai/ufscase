import pytrends
import pandas as pd
import time
import translators as ts
from pathlib import Path
import os
from datetime import date


countries = ['NL', 'DE', 'ES']
years = range(2010, 2021)


def yearToDates(year: int):
    return f"{year}-01-01 {year}-12-31"


def translate(text, country):
    """
    Translate the text to the languages originating from the country
    :param text: english text
    :param country: country abbreviation
    :return: list of translations
    """
    lang = country.lower()
    if lang == 'es':
        # Spanish, Basque, Catalan, Galician,
        langs = ['es', 'eu', 'ca', 'gl']
        translations = [ts.google(text, from_language='en', to_language=lang) for lang in langs]
    else:
        translations = [ts.google(text, from_language='en', to_language=lang)]
    return translations


def saveResult(df, file_name, folder_name=''):
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
        df.to_csv(folder_path + "/" + file_name + ".txt", header=None, index=None, sep = ', ', mode='a')
    else:
        try:
            os.mkdir(folder_path)
            df.to_csv(folder_path + "/" + file_name + ".txt", header=None, index=None, sep=', ', mode='a')
        except OSError:
            print("Failed to create folder, file saved in Data folder")
            df.to_csv(data_path + "/" + file_name + ".txt", header=None, index=None, sep = ', ', mode='a')




