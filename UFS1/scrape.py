import pandas as pd
import pytrends
import seaborn as sns
import matplotlib.pyplot as plt
from pytrends.request import TrendReq

pytrend = TrendReq(hl='nl')

KEYWORDS=['Asperge','Spruitjes','Pompoen'] 
KEYWORDS_CODES=[pytrend.suggestions(keyword=i)[0] for i in KEYWORDS] 
df_CODES= pd.DataFrame(KEYWORDS_CODES)
# print(df_CODES)

EXACT_KEYWORDS=df_CODES['mid'].to_list()
DATE_INTERVAL='2020-01-01 2021-01-01'
COUNTRY=["GB","DE", "NL"] #Use this link for iso country code: https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
CATEGORY=0 # Use this link to select categories: https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
SEARCH_TYPE='' #default is 'web searches',others include 'images','news','youtube','froogle' (google shopping)

Individual_EXACT_KEYWORD = list(zip(*[iter(EXACT_KEYWORDS)]*1))
Individual_EXACT_KEYWORD = [list(x) for x in Individual_EXACT_KEYWORD]
dicti = {}
i = 1
for Country in COUNTRY:
    for keyword in Individual_EXACT_KEYWORD:
        pytrend.build_payload(kw_list=keyword, 
                              timeframe = DATE_INTERVAL, 
                              geo = Country, 
                              cat=CATEGORY,
                              gprop=SEARCH_TYPE) 
        dicti[i] = pytrend.interest_over_time()
        i+=1
df_trends = pd.concat(dicti, axis=1)

df_trends.columns = df_trends.columns.droplevel(0) #drop outside header
df_trends = df_trends.drop('isPartial', axis = 1) #drop "isPartial"
df_trends.reset_index(level=0,inplace=True) #reset_index
df_trends.columns=['date','Asperge-GB', 'Spruitjes-GB', 'Pompoen-GB', 'Asperge-DE', 'Spruitjes-DE', 'Pompoen-DE', 'Asperge-NL', 'Spruitjes-NL', 'Pompoen-NL'] #change column names
# print(df_trends)

sns.set(color_codes=True)

# dx = sns.relplot(x="date", y=['Asperge-NL', 'Spruitjes-NL', 'Pompoen-NL'], kind="line", data=df_trends)
dx = df_trends.plot(x="date", y=['Asperge-NL', 'Spruitjes-NL', 'Pompoen-NL'], kind="line", title = "Dutch vegetables")
dx.set_xlabel('Date')
dx.set_ylabel('Trends Index')
dx.tick_params(axis='both', which='both', labelsize=10)
plt.show()