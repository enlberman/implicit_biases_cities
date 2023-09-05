from os.path import join

import numpy as np
import pandas
import matplotlib.pyplot as plt
import numpy
from scipy.stats import spearmanr, ttest_ind
from statsmodels.api import OLS,add_constant, Logit, categorical
from statsmodels.stats.multitest import multipletests
from met_brewer.palettes import met_brew
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
COLORS = met_brew('Morgenstern',n=6,brew_type="continuous")
STATE_FIPS = {"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69","PR":"72","VI":"78"}

data_path = 'IAT_data'
figure_path = 'IAT_figures'
iat_data = pandas.read_csv(join(data_path,'cbsa_iat.csv'))
cbsas_delineation = pandas.read_csv(join(data_path,'delineation_2020.csv',skiprows=0))
cbsas_delineation['FIPS State Code'] = cbsas_delineation['FIPS State Code'].astype(str).map(lambda x: x if len(x) == 2 else '0' + x)
cbsas_delineation['FIPS County Code'] = cbsas_delineation['FIPS County Code'].astype(str).map(
    lambda x: x if len(x) == 3 else ('0' + x) if len(x) == 2 else '00' + x)
cbsas_delineation['county'] = cbsas_delineation['FIPS State Code'].astype(str) + cbsas_delineation['FIPS County Code'].astype(str)


geo_colummn = 'GEO_ID'
cbsa_pop = pandas.read_csv(join(data_path,
        'ACSDT5Y%s.B01003_data_with_overlays.csv' % str(2019)))
cbsa_pop = cbsa_pop[cbsa_pop[geo_colummn].str.contains('US')]
cbsa_pop['cbsa_code'] = cbsa_pop[geo_colummn].map(lambda x: x.split('US')[1])

threshold = 500
suffixs = ['','_seg_idx','_gini','_exp']
A =[]
R = []
for suffix in suffixs:
    hom_corr = numpy.load(join(data_path,'hom_corr'+suffix+'.npy'),allow_pickle=True)
    white_pop = numpy.load(join(data_path,'white_pop'+suffix+'.npy'),allow_pickle=True)
    black_pop = numpy.load(join(data_path,'black_pop'+suffix+'.npy'),allow_pickle=True)
    white_hom = numpy.load(join(data_path,'white_hom'+suffix+'.npy'),allow_pickle=True)
    black_hom = numpy.load(join(data_path,'black_hom'+suffix+'.npy'),allow_pickle=True)
    cbsas = numpy.load(join(data_path,'cbsas'+suffix+'.npy'),allow_pickle=True)
    pops = numpy.load(join(data_path,'pops' + suffix + '.npy'), allow_pickle=True)
    years = numpy.unique(iat_data['year'])
    aics = []
    r2s = []
    for i in range(len(years)):
        print(years[i])
        year_data = iat_data[(iat_data['cbsa_code'].isin(cbsas[i])) & (iat_data['year'] == years[i])]
        y = year_data['iat_mean']
        pop = np.hstack([pops[i][cbsas[i] == x] for x in year_data['cbsa_code']])
        wh = np.hstack([white_hom[i][cbsas[i] == x] for x in year_data['cbsa_code']])
        bh = np.hstack([black_hom[i][cbsas[i] == x] for x in year_data['cbsa_code']])
        n = year_data['iat_n']
        keep = ~np.isnan(y) & (n>threshold)
        x = numpy.vstack([wh[keep]+bh[keep]]).T
        y = y[keep]
        pop = pop[keep]
        resid = OLS(numpy.log(y),add_constant(numpy.log(pop))).fit().resid
        f0 = OLS(resid, add_constant(x)).fit()
        f = [f0]
        for j in range(20):
            print(j)
            x = numpy.vstack([x.T,wh[keep]**(j+1)+bh[keep]**(j+1)]).T
            f.append(OLS(resid, add_constant(x)).fit())
        r2 = [t.rsquared_adj if t.f_pvalue<.05 else numpy.nan for t in f]
        aic = [t.aic if t.f_pvalue<.05 else numpy.nan for t in f]
        # plt.clf()
        # plt.plot(range(1, 22), r2, linestyle='--', alpha=0.5)
        # plt.xlabel('polynomial degree')
        # plt.xticks(range(1, 22, 2))
        # plt.ylabel(r'adjusted $R^2$')
        # ax = plt.twinx()
        # ax.plot(range(1, 22), aic)
        # ax.set_ylabel('AIC')
        # plt.tight_layout()
        # plt.show()
        r2s.append(r2)
        aics.append(numpy.nanmin(aic)-aic)

        # print()
    # print()
    A.append(numpy.vstack(aics))
    R.append(numpy.vstack(r2s))


aics = [list(filter(lambda x:~numpy.isnan(x),-numpy.vstack(A)[:,i])) for i in range(1,21)]
rs = [numpy.vstack(R)[:,i][numpy.vstack(R)[:,i]>0] for i in range(1,21)]
aips = numpy.vstack([ttest_ind(aics[0],aics[i]) for i in range(1,20)])
rps = numpy.vstack([ttest_ind(rs[0],rs[i]) for i in range(1,20)])
plt.clf()
nonsig = numpy.argwhere(aips[:,1]>.05).flatten()[-1]+2
b1 = plt.boxplot(aics[:nonsig])
b2 = plt.boxplot(aics[nonsig:],positions=range(nonsig+1,len(aics)+1))
[[x.set_alpha(0.5) for x in type] for type in b2.values()]
plt.plot([],[],alpha=1,label=r'$p = n.s.$',color='k')
plt.plot([],[],alpha=0.5,label=r'$p<.05$',color='k')
plt.legend()
plt.xlabel('polynomial degree')
plt.ylabel(r'$\Delta$ AIC')
plt.savefig(join(figure_path,'higher_order_polynomial_AIC.png'),dpi=300)


plt.clf()
nonsig = numpy.argwhere(rps[:,1]>.05).flatten()[-1]+2
b1 = plt.boxplot(rs[nonsig:],positions=range(nonsig+1,len(rs)+1))
b2 = plt.boxplot(rs[:nonsig],widths=0.5)
[[x.set_alpha(0.5) for x in type] for type in b2.values()]
plt.xlabel('polynomial degree')
plt.ylabel(r'adjusted $R^2$')
plt.savefig(join(figure_path,'higher_order_polynomial_R2.png'),dpi=300)













