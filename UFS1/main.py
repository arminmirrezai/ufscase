import ApiExtract
import pandas as pd

countries = ['NL', 'DE']
years = range(2016, 2021)
df = {}
for country in countries:
    df[country] = pd.concat(ApiExtract.extract(years, country))



