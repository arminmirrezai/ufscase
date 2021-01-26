from ApiExtract import extract
from Decompositions import Decompose
import Description

df = extract(range(2015, 2021), 'NL')
dd = Description.Data(df)
dmp = Decompose(df)

# Example trend
dmp.decompose_ma('tosti')
print(f"Strenght of seasonality {dmp.seasonality_F()}, Strenght of trend {dmp.trend_F()}")

# main changed