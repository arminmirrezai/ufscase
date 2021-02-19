from ApiExtract import extract
from Decompositions import Decompose
import Description
import time

df = extract(range(2016, 2021), 'NL')
dd = Description.Data(df)
dmp = Decompose(df)

start_time = time.time()
for kw in df.keyword.unique():
    dmp.decompose_robustSTL(kw)
print(f"Robust STL took in total {time.time() - start_time} seconds")
