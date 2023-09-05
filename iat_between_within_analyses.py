import numpy as np
import pandas
import matplotlib.pyplot as plt
import numpy
from statsmodels.api import OLS,add_constant, Logit, categorical
from met_brewer.palettes import met_brew
from os.path import join

STATE_FIPS = {"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69","PR":"72","VI":"78"}

data_path = 'IAT_data'

iat_data = pandas.read_csv(join(data_path,'cbsa_iat.csv'))

threshold = 500
suffixs = ['']
ff1 = []
ff2 = []
ff3 = []
ff4 = []
for suffix in suffixs:
    white_pop = numpy.load(join(data_path,'white_pop'+suffix+'.npy'),allow_pickle=True)
    black_pop = numpy.load(join(data_path,'black_pop'+suffix+'.npy'),allow_pickle=True)
    white_hom = numpy.load(join(data_path,'white_hom' + suffix + '.npy'), allow_pickle=True)
    black_hom = numpy.load(join(data_path,'black_hom' + suffix + '.npy'), allow_pickle=True)

    cbsas = numpy.load(join(data_path,'cbsas'+suffix+'.npy'),allow_pickle=True)
    pops = numpy.load(join(data_path,'pops' + suffix + '.npy'), allow_pickle=True)
    years = numpy.unique(iat_data['year'])
    f1s = []
    f2s = []
    f3s = []
    f4s = []
    for i in range(len(years)):
        print(years[i])
        year_data = iat_data[(iat_data['cbsa_code'].isin(cbsas[i])) & (iat_data['year'] == years[i])]
        y = year_data['iat_mean']
        pop = np.hstack([pops[i][cbsas[i] == x] for x in year_data['cbsa_code']])
        wp = np.hstack([white_pop[i][cbsas[i] == x] for x in year_data['cbsa_code']])
        bp = np.hstack([black_pop[i][cbsas[i] == x] for x in year_data['cbsa_code']])
        wh = np.hstack([white_hom[i][cbsas[i] == x] for x in year_data['cbsa_code']])
        bh = np.hstack([black_hom[i][cbsas[i] == x] for x in year_data['cbsa_code']])
        n = year_data['iat_n']
        keep = ~np.isnan(y) & (n>threshold)
        b1 = []
        b2 = []
        b3 = []
        b4 = []
        f3 = OLS(numpy.log(y[keep]),
                 add_constant(numpy.vstack([numpy.log(wp - wp ** 2),
                                            numpy.log(wh + bh),
                                            numpy.log(wp ** 2 * (wh - bh) + (1 - wh) * (1 + bh))])[:,
                              keep].T)).fit()
        f4 = OLS(numpy.log(y[keep]),
                 add_constant(numpy.vstack([numpy.log(wp - wp ** 2),
                                            numpy.log(wh + bh),
                                            ])[:, keep].T)).fit()
        print(f3.summary())
        print(f4.summary())
        print('##############################################')

        for i in range(500):
            id = numpy.random.choice(range(len(wp)),len(wp),replace=True)
            wpr = wp[id]
            bpr = bp[id]
            whr = wh[id]
            bhr = bh[id]
            f1 = OLS(numpy.log(y[keep]), add_constant(numpy.log((wpr - wpr ** 2)[keep]))).fit()
            f2 = OLS(numpy.log(y[keep]),add_constant(numpy.log(((wpr - wpr ** 2) / (1 - 2 * (wpr - wpr ** 2)))[keep]))).fit()
            f3 = OLS(numpy.log(y[keep]),
                     add_constant(numpy.vstack([numpy.log(wpr - wpr ** 2),
                                                numpy.log(whr + bhr),
                                                numpy.log(wpr ** 2 * (whr - bhr) + (1 - whr) * (1 + bhr))])[:,
                                  keep].T)).fit()
            f4 = OLS(numpy.log(y[keep]),
                     add_constant(numpy.vstack([numpy.log(wpr - wpr ** 2),
                                                numpy.log(whr + bhr),
                                                ])[:, keep].T)).fit()
            b1.append([f1.rsquared,f1.aic])
            b2.append([f2.rsquared, f2.aic])
            b3.append([f3.rsquared_adj, f3.aic])
            b4.append([f4.rsquared_adj, f4.aic])

        f1s.append(b1)
        f2s.append(b2)
        f3s.append(b3)
        f4s.append(b4)

    ff1.append(f1s)
    ff2.append(f2s)
    ff3.append(f3s)
    ff4.append(f4s)
        # print()
    # print()


plt.clf()
plt.axhline(0,color='k')
plt.violinplot([numpy.vstack(ff1[0][i])[:,1]-numpy.vstack(ff2[0][i])[:,1] for i in range(11)],showextrema=False,
    quantiles=[[0.025,0.975]for i in range(11)])
plt.ylabel(r'$\Delta\ AIC$')
plt.xticks([1,6,11],[2010,2015,2020])
plt.tight_layout()
plt.savefig(join(data_path,'between_within_delta_aic.png'),dpi=300)
plt.clf()
plt.axhline(0,color='k')
plt.violinplot([numpy.vstack(ff1[0][i])[:,0]-numpy.vstack(ff2[0][i])[:,0] for i in range(11)],showextrema=False,
               quantiles=[[0.025,0.975]for i in range(11)])
plt.ylabel(r'$\Delta\ R^2$')
plt.xticks([1,6,11],[2010,2015,2020])
plt.tight_layout()
plt.savefig(join(data_path,'between_within_delta_r2.png'),dpi=300)

plt.clf()
plt.axhline(0,color='k')
plt.violinplot([numpy.vstack(ff4[0][i])[:,1]-numpy.vstack(ff3[0][i])[:,1] for i in range(11)],showextrema=False,
               quantiles=[[0.025,0.975]for i in range(11)])
plt.ylabel(r'$\Delta\ AIC$')
plt.xticks([1,6,11],[2010,2015,2020])
plt.tight_layout()
plt.savefig(join(data_path,'between_within_delta_aic_full.png'),dpi=300)
plt.clf()
plt.axhline(0,color='k')
plt.violinplot([numpy.vstack(ff4[0][i])[:,0]-numpy.vstack(ff3[0][i])[:,0] for i in range(11)],showextrema=False,
               quantiles=[[0.025,0.975]for i in range(11)])
plt.ylabel(r'$\Delta\ R^2$')
plt.xticks([1,6,11],[2010,2015,2020])
plt.tight_layout()
plt.savefig(join(data_path,'between_within_delta_r2_full.png'),dpi=300)
