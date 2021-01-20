import ApiExtract
import pandas as pd


if __name__ == "__main__":
    countries = ['NL', 'DE']
    years = range(2016, 2021)
    frames = []
    for country in countries:
        frames += ApiExtract.extract(years, country)

    df = pd.concat(frames)
    groups = df.groupby('category')['keyword'].agg('unique')
    print(df)
    print(groups)
