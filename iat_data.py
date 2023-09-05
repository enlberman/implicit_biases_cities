import pickle
from os.path import join

import numpy as np
import numpy.random
import pandas
import matplotlib.pyplot as plt
STATE_FIPS = {"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69","PR":"72","VI":"78"}


cols_of_interest = [
    "D_biep.White_Good_all", # overall iat score
    "birthyear",
    "num_002", # number of iats complete
    "raceomb_002",
    "raceombmulti",
    "ethnicityomb",
    "birthsex",
    "sex",
    "edu_14",
    "politicalid_7",
    "religionid",
    "user_id",
    "date",

]

data_path = 'IAT_data'

slave_data = pandas.read_csv(join(data_path,'slaves_1860_job_dataverse/abs-jop-cces-white-countydata.csv'))

cbsas_delineation = pandas.read_csv(join(data_path,'delineation_2020.csv'),skiprows=0)
cbsas_delineation['FIPS State Code'] = cbsas_delineation['FIPS State Code'].astype(str).map(lambda x: x if len(x) == 2 else '0' + x)
cbsas_delineation['FIPS County Code'] = cbsas_delineation['FIPS County Code'].astype(str).map(
    lambda x: x if len(x) == 3 else ('0' + x) if len(x) == 2 else '00' + x)
cbsas_delineation['county'] = cbsas_delineation['FIPS State Code'].astype(str) + cbsas_delineation['FIPS County Code'].astype(str)
cbsas_delineation['CBSA Code'] = cbsas_delineation['CBSA Code'].astype(int)

years = list(range(2010,2021))
data = []
halves = []
for year in years:
    print(year)
    year_halves = []
    iat_data = pandas.read_csv(join(data_path,'race_iat_geo_data_%d.csv' % year))
    for cbsa in np.unique(cbsas_delineation['CBSA Code']):
        cbsa_df = cbsas_delineation[cbsas_delineation['CBSA Code']==cbsa]
        cbsa_iat = iat_data[iat_data['COUNTY'].astype(str).isin(cbsa_df['county'])]
        cbsa_iat = cbsa_iat[cbsa_iat['D_biep.White_Good_all']!=' ']
        cbsa_slave_data = slave_data[slave_data['fips'].astype(str).isin(cbsa_df['county'])][['pslave1860', 'totpop1860']]
        cbsa_pslave = (cbsa_slave_data.values[:,0]*cbsa_slave_data.values[:,1]).sum()/cbsa_slave_data.values[:,1].sum()
        uncertainty_cols = list(filter(lambda x: x.__contains__('sius'),list(cbsa_iat.columns)))
        uncertainty_scores = cbsa_iat.copy()
        for c in uncertainty_cols:
            uncertainty_scores = uncertainty_scores[uncertainty_scores[c]!=' ']
        uncertainty_scores = uncertainty_scores[uncertainty_cols]
        uncertainty_scores = uncertainty_scores.values.astype(int).sum(1)
        death_cols = list(filter(lambda x: x.__contains__('death'), list(cbsa_iat.columns)))
        death = cbsa_iat.copy()
        for c in death_cols:
            death = death[death[c] != ' ']
        death = death[death_cols]
        death = death.values.astype(int).sum(1)
        data.append(
            [
                year,
                cbsa,
                cbsa_iat['D_biep.White_Good_all'].astype(float).mean(),
                cbsa_iat['D_biep.White_Good_all'].astype(float).std(),
                cbsa_iat.shape[0],
                uncertainty_scores.mean(),
                uncertainty_scores.std(),
                uncertainty_scores.shape[0],
                death.mean(),
                death.std(),
                death.shape[0],
                cbsa_pslave
             ]
        )
        d = cbsa_iat['D_biep.White_Good_all'].astype(float)
        cbsa_halves = []
        for s in range(500):
            nn = len(d)
            order = numpy.random.permutation(range(nn))
            half = nn//2
            if nn>10:
                cbsa_halves.append([d.values[order][:half].mean(), d.values[order][half:].mean()])
            else:
                cbsa_halves.append([np.nan,np.nan])
        year_halves.append([cbsa,cbsa_halves])
    print('done with %d' % year)
    halves.append(year_halves)

with open(join(data_path,'cbsa_iat_split_halves.pkl'),'wb') as f:
    pickle.dump(halves,f)

cbsa_data = pandas.DataFrame(data,columns=['year',
                                           'cbsa_code',
                                           'iat_mean',
                                           'iat_std',
                                           'iat_n',
                                           'uncertainty_mean',
                                           'uncertainty_std',
                                           'uncertainty_n',
                                           'death_mean',
                                           'death_std',
                                           'death_n',
                                           'pslave_1860'
                                           ])
cbsa_data.to_csv(join(data_path,'cbsa_iat.csv'))
