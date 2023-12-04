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
from statsmodels.stats.outliers_influence import variance_inflation_factor
COLORS = met_brew('Morgenstern',n=6,brew_type="continuous")
STATE_FIPS = {"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69","PR":"72","VI":"78"}

data_path = 'IAT_data'
figure_path = 'IAT_figures'
white_pop = numpy.load(join(data_path,'white_pop.npy'),allow_pickle=True)
black_pop = numpy.load(join(data_path,'black_pop.npy'),allow_pickle=True)
cbsas = numpy.load(join(data_path,'cbsas.npy'),allow_pickle=True)

years = range(2010,2021)
spearmans = []
for j in range(len(years)):
    year = years[j]
    data = pandas.read_csv(join(data_path,'cbsa_iat_individual_%d.csv' % year))
    race_col = 'raceomb_002' if year>=2016 else 'raceomb'
    data = data[data['cbsa_code'].astype(int).isin(cbsas[j].values)]
    n_individual_total = len(data)
    data = data[data[race_col].astype(str)!=' ']
    data = data[data[race_col].astype(str) != 'nan']

    data['white'] = data[race_col].astype(str).map(lambda x: 1 if x=='6' else 0).astype(bool)
    data['black'] = data[race_col].astype(str).map(lambda x: 1 if x == '5' else 0).astype(bool)
    data = data[['cbsa_code','white','black']]

    n_individuals = len(data)
    data = data.groupby('cbsa_code').mean().reset_index()

    idxs = [numpy.argwhere(cbsas[j].values == x).flatten() for x in data['cbsa_code'].astype(int)]
    data['white_pop'] = [white_pop[j].values[t[0]] if len(t) > 0 else numpy.nan for t in idxs]
    data['black_pop'] = [black_pop[j].values[t[0]] if len(t) > 0 else numpy.nan for t in idxs]

    plt.clf()
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    x = data['white']
    y = data['white_pop']
    rs = spearmanr(x[~numpy.isnan(x) & ~numpy.isnan(y)], y[~numpy.isnan(x) & ~numpy.isnan(y)])
    plt.scatter(x, y, label=r'$r_s=%.2f; p<0.001$' % (rs[0]))
    plt.xlabel('iat sample % white')
    plt.ylabel('cbsa population % white')
    plt.legend()
    plt.subplot(1, 2, 2)
    x = data['black']
    y = data['black_pop']
    rs2 = spearmanr(x[~numpy.isnan(x) & ~numpy.isnan(y)], y[~numpy.isnan(x) & ~numpy.isnan(y)])
    plt.scatter(x, y, label=r'$r_s=%.2f; p<0.001$' % (rs2[0]))
    plt.legend()
    plt.xlabel('iat sample % black')
    plt.ylabel('cbsa population % black')
    plt.tight_layout()
    print(len(x))
    plt.savefig(join(figure_path,'Race_comparison_figure_%s.png' % str(year)), dpi=300)
    spearmans.append([year,'%.2f'% rs[0],'%.2e'% rs[1] if rs[1]>=0.001 else '<0.001','%.2f' % rs2[0],
                      '%.2e' % rs2[1] if rs2[1]>=0.001 else '<0.001',len(x)])\
        #,n_individuals, n_individual_total])

spearmans = numpy.vstack(spearmans)
df = pandas.DataFrame(spearmans,columns=[
    'year','$r_s$ (white)','p value',r'$r_s$ (black)','p value','# cbsas'])
with open(join(figure_path,'race_comparison.txt'),'w') as f:
    f.write(df.to_latex(index=False))
print()
