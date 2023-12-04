from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas
from os.path import join
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
STATE_FIPS = {"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69","PR":"72","VI":"78"}


cols_of_interest = [
    "D_biep.White_Good_all", # overall iat score
    "birthyear",
    "num_002", # number of iats complete
    "raceomb_002",
    "raceomb002",
    "raceomb",
    "birthsex",
    "birthSex",
    "edu_14",
    "politicalid_7",
    "religionid",
    "user_id",
    "date",

]

iat_race_cols = [    "raceomb_002",
    "raceomb002",
    "raceomb",]
birthsex_cols = ["birthsex","birthSex"]

data_path = 'IAT_data'


cbsas_delineation = pandas.read_csv(join(data_path,'delineation_2020.csv'),skiprows=0)#skiprows=2
cbsas_delineation['FIPS State Code'] = cbsas_delineation['FIPS State Code'].astype(str).map(lambda x: x if len(x) == 2 else '0' + x)
cbsas_delineation['FIPS County Code'] = cbsas_delineation['FIPS County Code'].astype(str).map(
    lambda x: x if len(x) == 3 else ('0' + x) if len(x) == 2 else '00' + x)
cbsas_delineation['county'] = cbsas_delineation['FIPS State Code'].astype(str) + cbsas_delineation['FIPS County Code'].astype(str)

years = list(range(2010,2021))
data=[]
for year in years:
    print(year)
    iat_data = pandas.read_csv(join(data_path,'race_iat_geo_data_%d.csv' % year))
    iat_data['raceomb'] = iat_data[iat_race_cols[np.argwhere([len(np.unique(iat_data[c].values.astype(str)))>1 for c in iat_race_cols]).flatten()[0]]]
    birthsex_col = np.argwhere([len(np.unique(iat_data[c].values.astype(str)))>1 for c in birthsex_cols]).flatten()
    birthsex_col = birthsex_col.flatten()[0] if len(birthsex_col) > 0 else 0
    iat_data['birthsex'] = iat_data[birthsex_cols[birthsex_col]]

    year = str(int(year))
    race_eth = pandas.read_csv(join(data_path,'ACSDT5Y%s.B02001_data_with_overlays.csv') % year)
    total_column = 'B02001_001E'
    white_column = 'B02001_002E'
    black_column = 'B02001_003E'
    native_column = 'B02001_004E'
    asian_column = 'B02001_005E'
    island_column = 'B02001_006E'
    other_column = 'B02001_007E'
    eth_cols = [white_column, black_column, native_column, asian_column, island_column, other_column]
    geo_colummn = 'GEO_ID'
    race_eth = race_eth[race_eth[geo_colummn].str.contains('US')]
    race_eth['county'] = race_eth[geo_colummn].map(lambda x: x.split('US')[1][:5])

    cbsa_pop = pandas.read_csv(join(data_path,'ACSDT5Y%s.B01003_data_with_overlays.csv' % year))
    cbsa_pop = cbsa_pop[cbsa_pop[geo_colummn].str.contains('US')]
    cbsa_pop['cbsa_code'] = cbsa_pop[geo_colummn].map(lambda x: x.split('US')[1])

    cbsas = pandas.read_csv(join(data_path,'delineation_2020.csv'),skiprows=0)#skiprows=2
    cbsas['FIPS State Code'] = cbsas['FIPS State Code'].astype(str).map(lambda x: x if len(x)==2 else '0'+x)
    cbsas['FIPS County Code'] = cbsas['FIPS County Code'].astype(str).map(lambda x: x if len(x)==3 else ('0'+ x) if len(x)==2 else '00'+x)
    cbsas['county'] =  cbsas['FIPS State Code'].astype(str) + cbsas['FIPS County Code'].astype(str)
    cbsa_codes = np.unique(cbsas['CBSA Code'])

    joined_race_eth = race_eth.set_index('county').join(cbsas.set_index('county')).reset_index()

    def get_cbsa_data(cbsa):
        cbsa_df = cbsas_delineation[cbsas_delineation['CBSA Code'] == cbsa]
        cbsa_iat = iat_data[iat_data['COUNTY'].astype(str).isin(cbsa_df['county'])]
        cbsa_iat = cbsa_iat[cbsa_iat['D_biep.White_Good_all'] != ' ']

        if len(cbsa_iat) == 0:
            return None
        cbsa_race_eth = joined_race_eth[joined_race_eth['CBSA Code'] == cbsa]
        ratios = np.array([
            cbsa_race_eth[x].astype(float).sum() / cbsa_race_eth[total_column].astype(float).sum()
            for x in eth_cols
        ])
        tot = cbsa_race_eth[total_column].astype(float)
        tot = tot.map(lambda x: x if x > 0 else np.nan)
        local_ratio = [
            cbsa_race_eth[x].astype(float) / tot for x in eth_cols
        ]
        homophily = []
        seggregation_index = []
        gini_index = []
        inter_exposure = []
        for i in range(len(eth_cols)):
            homophily.append(np.abs(
                local_ratio[i] - ratios[i]
            ).mean())
            seggregation_index.append(np.nansum(np.abs(
                (local_ratio[i] - ratios[i]) * tot
            )) / (
                                              2 * np.nansum(tot) * ratios[i] * (1 - ratios[i])
                                      )
                                      )
            gini_index.append(
                np.nansum([
                    [np.abs(local_ratio[i].values[j] - local_ratio[i].values[k]) * tot.values[j] * tot.values[k]
                     for j in range(len(local_ratio[i]))] for k in range(len(local_ratio[i]))]) /
                (2 * np.nansum(tot) ** 2 * ratios[i] * (1 - ratios[i]))
            )
            inter_exposure.append(np.nansum(tot * local_ratio[i] ** 2)
                                  / (np.nansum(local_ratio[i] * tot) * (1 - ratios[i]))
                                  - ratios[i] / (1 - ratios[i]))

        heterophobia_adjustment = (homophily[0] if ~np.isnan(homophily[0]) else 0) + \
                                  (homophily[1] if ~np.isnan(homophily[1]) else 0)

        heterophobia_adjustment_seg = (seggregation_index[0] if ~np.isnan(seggregation_index[0]) else 0) + \
                                      (seggregation_index[1] if ~np.isnan(seggregation_index[1]) else 0)

        heterophobia_adjustment_gini = \
            (gini_index[0] if ~np.isnan(gini_index[0]) else 0) + \
            (gini_index[1] if ~np.isnan(gini_index[1]) else 0)

        heterophobia_adjustment_exp = \
            (inter_exposure[0] if ~np.isnan(inter_exposure[0]) else 0) + \
            (inter_exposure[1] if ~np.isnan(inter_exposure[1]) else 0)

        maj_group_adjustment = np.log(ratios[0] - ratios[0] ** 2)

        if cbsa_iat.shape[0] > 500:
            pop_shape = cbsa_pop[cbsa_pop['cbsa_code'] == str(cbsa)].shape[0]
            cbsa_iat['cbsa_code'] = np.repeat(cbsa, cbsa_iat.shape[0])
            cbsa_iat['cbsa_pop'] = np.repeat(
                cbsa_pop[cbsa_pop['cbsa_code'] == str(cbsa)]['B01003_001E'].values[0] if pop_shape > 0 else np.nan,
                cbsa_iat.shape[0])
            cbsa_iat['maj_group_adjust'] = np.repeat(maj_group_adjustment, cbsa_iat.shape[0])
            cbsa_iat['het_correction'] = np.repeat(heterophobia_adjustment, cbsa_iat.shape[0])
            cbsa_iat['het_correction_seg'] = np.repeat(heterophobia_adjustment_seg, cbsa_iat.shape[0])
            cbsa_iat['het_correction_exp'] = np.repeat(heterophobia_adjustment_exp, cbsa_iat.shape[0])
            cbsa_iat['het_correction_gini'] = np.repeat(heterophobia_adjustment_gini, cbsa_iat.shape[0])
            return cbsa_iat
        else:
            return None
    # now parallelize
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(get_cbsa_data, np.unique(cbsas_delineation['CBSA Code']))
    res = [x for x in results]
    data = data + list(filter(lambda x: x is not None,res))
    pandas.concat(data).to_csv(join(data_path,'cbsa_iat_individual_%s.csv' % year))

    print('done with %s' % year)

