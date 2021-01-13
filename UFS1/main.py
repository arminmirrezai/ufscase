import ApiExtract

countries = ['NL', 'DE']
years = range(2017, 2021)

for country in countries:
    data = ApiExtract.extract(years, country)
