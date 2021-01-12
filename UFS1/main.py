import ApiExtract
import SearchData

countries = ['NL', 'DE', 'ES']
years = range(2017, 2021)

data = ApiExtract.extract(years, countries[0])
