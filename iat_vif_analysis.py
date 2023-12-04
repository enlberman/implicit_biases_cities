from os.path import join

import numpy as np
import pandas
import matplotlib.pyplot as plt
import numpy
from scipy.stats import spearmanr
from statsmodels.api import OLS,add_constant, Logit, categorical
from statsmodels.stats.multitest import multipletests
# from met_brewer.palettes import met_brew
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
# COLORS = met_brew('Morgenstern',n=6,brew_type="continuous")
STATE_FIPS = {"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69","PR":"72","VI":"78"}

data_path = 'IAT_data'
figure_path = 'IAT_figures'

iat_data = pandas.read_csv(join(data_path,'cbsa_iat.csv'))


suffixs = ['','_seg_idx','_gini','_exp']
for suffix in suffixs:
    white_pop = numpy.load(join(data_path,'white_pop'+suffix+'.npy'),allow_pickle=True)
    black_pop = numpy.load(join(data_path,'black_pop'+suffix+'.npy'),allow_pickle=True)
    white_hom = numpy.load(join(data_path,'white_hom'+suffix+'.npy'),allow_pickle=True)
    black_hom = numpy.load(join(data_path,'black_hom'+suffix+'.npy'),allow_pickle=True)
    cbsas = numpy.load(join(data_path,'cbsas'+suffix+'.npy'),allow_pickle=True)

    demo_df = pandas.DataFrame({'cbsa_code':cbsas[-2].astype(int).astype(str),'white_pop':white_pop[-2],'black_pop':black_pop[-2],'white_hom':white_hom[-2],'black_hom':black_hom[-2],
                                'iat_n':[t.values[0] if len(t)>0 else numpy.nan for t in [iat_data[(iat_data['cbsa_code']==c) & (iat_data['year']==2018)]['iat_n'] for c in cbsas[-2].values]]
                                })
    # demo_df = demo_df[demo_df['iat_n']>500]
    x=(demo_df['white_pop'] - demo_df['white_pop'] ** 2)
    y=demo_df['white_hom']+demo_df['black_hom']
    print(spearmanr(y[~numpy.isnan(y)], demo_df['white_pop'].values[~numpy.isnan(y)]))
    x = x[~numpy.isnan(y)]
    y = y[~numpy.isnan(y)]

    vif = variance_inflation_factor(numpy.vstack((x,y)).T,[0])
    print(vif)
    plt.clf()
    print(spearmanr(x,y))
    plt.scatter(x,y,alpha=.5)
    plt.xlabel(r'Diversity ( $\frac{N_g}{N}-\left (\frac{N_g}{N}\right)^2$)')
    plt.ylabel(r'$s_w+s_b$')
    plt.text(.05, (y.min()*1.1 if suffix!='' else .2) if suffix!='_exp' else.45 ,'VIF=%.2f' % vif,size=16)
    plt.tight_layout()
    plt.savefig(join(figure_path,'VIF_figure%s.png' % suffix),dpi=300)

print()













