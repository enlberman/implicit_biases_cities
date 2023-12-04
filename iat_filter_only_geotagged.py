from os.path import join

import numpy as np
import pandas
import matplotlib.pyplot as plt
STATE_FIPS = {"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69","PR":"72","VI":"78"}


cols_of_interest = [
    "D_biep.White_Good_all", # overall iat score
    "birthyear",
    "num_002", # number of iats complete
    "raceomb_002",
    "raceomb002",
    "raceomb",
    "raceombmulti",
    "ethnicityomb",
    "birthSex",
    "birthsex",
    "sex",
    "edu_14",
    "politicalid_7",
    "religionid",
    "user_id",
    "date",

]

data_path = 'IAT_data'
data_path = '/home/andrewstier/Downloads/IAT_data'

years = list(range(2010,2021))
for year in years:
    print(year)
    c=0
    dfs = []
    iat_data = pandas.read_csv(join(data_path,'Race IAT.public.%d.csv' % year), chunksize=90000,encoding='latin')
    for df in iat_data:
        c+=1
        print(c)
        df = df[df['STATE']!=' ']
        df = df[df['STATE'].isin(STATE_FIPS.keys())]
        df['COUNTY'] = df['STATE'].map(lambda x: STATE_FIPS[x]) + df['CountyNo']
        cols = ['COUNTY']+cols_of_interest + list(filter(lambda x: x.__contains__('sius'),list(df.columns)))+list(filter(lambda x: x.__contains__('deathanx'),list(df.columns)))
        for col in cols:
            if not list(df.columns).__contains__(col):
                df[col] = np.nan
        dfs.append(df[cols])

    dfs = pandas.concat(dfs)
    dfs.to_csv(join(data_path,'race_iat_geo_data_%d.csv' % year))
