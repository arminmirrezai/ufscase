import pytrends
import pandas as pd
import time
import translators as ts


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

