import pickle
from os.path import join

import numpy as np
import pandas
import matplotlib.pyplot as plt
import numpy
from scipy.stats import spearmanr
from statsmodels.api import OLS,add_constant, Logit, categorical
from statsmodels.stats.multitest import multipletests
from met_brewer.palettes import met_brew
from scipy.stats import pearsonr
COLORS = met_brew('Morgenstern',n=6,brew_type="continuous")
STATE_FIPS = {"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69","PR":"72","VI":"78"}

data_path = 'IAT_data'
figure_path = 'IAT_figures'
iat_data = pandas.read_csv(join(data_path,'cbsa_iat.csv'))

with open(join(data_path,'cbsa_iat_split_halves.pkl'),'rb') as f:
    split_halves = pickle.load(f)

heat_data = pandas.read_csv(join(data_path,'North America Land Data Assimilation System (NLDAS) Daily Air Temperatures and Heat Index (1979-2011).txt'), sep='\t')
heat_data = heat_data[heat_data['Notes']=='Total']
heat_data = heat_data[~numpy.isnan(heat_data['County Code'])]
heat_data['county'] = heat_data['County Code'].astype(int).astype(str)
heat_data['county'] = heat_data['county'].map(lambda x: x if len(x)==5 else '0'+x)
heat_data = heat_data[['county', 'Avg Daily Max Heat Index (C)']]
heat_data = heat_data[heat_data['Avg Daily Max Heat Index (C)']!='Missing']
heat_data['Avg Daily Max Heat Index (C)'] = heat_data['Avg Daily Max Heat Index (C)'].astype(float)

adi_data = pandas.read_csv(join(data_path,'US_2019_ADI_Census Block Group_v3.1.txt'))
adi_data['county'] = adi_data['FIPS'].astype(str).map(lambda x: x[-4:]).astype(int)
adi_data = adi_data[~adi_data['ADI_NATRANK'].isin(['GQ','GQ-PH','NONE','PH','QDI'])]
adi_data['ADI_NATRANK'] = adi_data['ADI_NATRANK'].astype(float)
adi_data['county'] = adi_data['county'].astype(int).astype(str)
adi_data['county'] = adi_data['county'].map(lambda x: x if len(x)==5 else '0'+x)
adi_data = adi_data[['county','ADI_NATRANK']].groupby('county').mean()

cbsas_delineation = pandas.read_csv(join(data_path,'delineation_2020.csv'),skiprows=0)
cbsas_delineation['FIPS State Code'] = cbsas_delineation['FIPS State Code'].astype(str).map(lambda x: x if len(x) == 2 else '0' + x)
cbsas_delineation['FIPS County Code'] = cbsas_delineation['FIPS County Code'].astype(str).map(
    lambda x: x if len(x) == 3 else ('0' + x) if len(x) == 2 else '00' + x)
cbsas_delineation['county'] = cbsas_delineation['FIPS State Code'].astype(str) + cbsas_delineation['FIPS County Code'].astype(str)

joined_heat = heat_data.set_index('county').join(cbsas_delineation.set_index('county'), rsuffix='_')[['Avg Daily Max Heat Index (C)','CBSA Code']
].groupby('CBSA Code').mean().reset_index()

joined_adi = adi_data.join(cbsas_delineation.set_index('county'), rsuffix='_')[['ADI_NATRANK','CBSA Code']
].groupby('CBSA Code').mean().reset_index()

geo_colummn = 'GEO_ID'
cbsa_pop = pandas.read_csv(join(data_path,'ACSDT5Y%s.B01003_data_with_overlays.csv' % str(2019)))
cbsa_pop = cbsa_pop[cbsa_pop[geo_colummn].str.contains('US')]
cbsa_pop['cbsa_code'] = cbsa_pop[geo_colummn].map(lambda x: x.split('US')[1])

joined_heat['CBSA Code'] = joined_heat['CBSA Code'].astype(int).astype(str)
joined_heat = joined_heat.set_index('CBSA Code').join(cbsa_pop.set_index('cbsa_code'))

suffix = '_exp'
hom_corr = numpy.load(join(data_path,'hom_corr'+suffix+'.npy'),allow_pickle=True)
white_pop = numpy.load(join(data_path,'white_pop'+suffix+'.npy'),allow_pickle=True)
black_pop = numpy.load(join(data_path,'black_pop'+suffix+'.npy'),allow_pickle=True)
white_hom = numpy.load(join(data_path,'white_hom'+suffix+'.npy'),allow_pickle=True)
black_hom = numpy.load(join(data_path,'black_hom'+suffix+'.npy'),allow_pickle=True)
cbsas = numpy.load(join(data_path,'cbsas'+suffix+'.npy'),allow_pickle=True)

demo_df = pandas.DataFrame({'cbsa_code':cbsas[-2].astype(int).astype(str),'hom_corr':hom_corr[-2],'white_pop':white_pop[-2],'black_pop':black_pop[-2],'white_hom':white_hom[-2],'black_hom':black_hom[-2]})
joined_heat = joined_heat.reset_index().set_index('CBSA Code').join(demo_df.set_index('cbsa_code')).reset_index()

plt.clf()
y = joined_heat['Avg Daily Max Heat Index (C)'].astype(float).values
x = joined_heat['black_pop'].astype(float).values
# x = x-x**2
x = x[~numpy.isnan(y)]
y = y[~numpy.isnan(y)]
y = y[~numpy.isnan(x)]
x = x[~numpy.isnan(x)]
plt.scatter(x,y,alpha=.5)
plt.xscale('log')
f = OLS(y,add_constant(numpy.log(x))).fit()
rs = spearmanr(x[~numpy.isnan(x)],y[~numpy.isnan(x)])
plt.plot(x[numpy.argsort(x)],f.fittedvalues[numpy.argsort(x)],color='k',linestyle='--')
plt.text(.105,29,r'$r_s=%.2f$'%rs[0]+'\n'+r'$p=%.2e$' % rs[1] if rs[1]>= 0.001 else r"$p<0.001$",size=15)
plt.xlabel('% Black')
plt.ylabel('Average Daily Max Heat Index (C)')
plt.savefig(join(figure_path,'heat_black.png'),dpi=300)










