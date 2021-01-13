import pandas as pd
from pathlib import Path
import translators as ts
import os


class GSData:

    def __init__(self):
        self.df = None
        self.df_kws = None

    def load_data_UF(self):
        """
        Extract the key word data
        :return: pandas data frame
        """
        data_path = Path.cwd().absolute().parents[0].as_posix() + "/Data/case_study_trends_data_20210107.csv"
        self.df = pd.read_csv(filepath_or_buffer=data_path, sep=',', header=0)
        self.df = self.df.astype({'date': 'datetime64', 'startDate': 'datetime64', 'endDate': 'datetime64'})
        return self.df

    def load_key_words(self, country, translated=True):
        """
        Load the key words for a specific country
        :param country: country of set {NL, DE, ES}
        :param translated: True if you also want the english word
        :return: data frame with keywords
        """
        folder_path = Path.cwd().absolute().parents[0].as_posix() + "/Data/KeyWords"
        file_name = country + ('_EN' if translated else '') + '.txt'
        file_path = folder_path + "/" + file_name
        if os.path.exists(file_path):
            self.df_kws = pd.read_csv(file_path, index_col=0)
        else:
            self._create_folder(folder_path)
            key_words = self._key_words_UF(country)
            if translated:
                key_words_trans = [[key_word, ts.google(key_word, from_language=country.lower(), to_language='en')]
                                    for key_word in key_words]
                self.df_kws = pd.DataFrame(key_words_trans)
                self.df_kws.columns = [country, 'EN']
            else:
                self.df_kws = pd.DataFrame(key_words)
                self.df_kws.columns = [country]

            self.df_kws.to_csv(file_path, sep=',', mode='w')
        return self.df_kws

    def _key_words_UF(self, country=''):
        """
        Extract the keywords given by unilever food
        :param country: country of set {NL, DE, ES}
        :return: key words countries
        """
        if self.df is None: self.load_data_UF()
        if country != '':
            return self.df[self.df['countryCode'] == country]['keyword'].unique()
        else:
            return self.df.groupby('countryCode')['keyword'].unique()

    @staticmethod
    def _create_folder(path):
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except OSError:
                print("Failed to create folder")

    @staticmethod
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

