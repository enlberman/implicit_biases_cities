import pickle
from os.path import join

import numpy as np
import pandas
import matplotlib.pyplot as plt
import numpy
from statsmodels.api import OLS,add_constant, Logit, categorical
from statsmodels.stats.multitest import multipletests
from met_brewer.palettes import met_brew
from scipy.stats import pearsonr
COLORS = met_brew('Morgenstern',n=6,brew_type="continuous")
STATE_FIPS = {"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69","PR":"72","VI":"78"}

# Set up data paths
figure_root_path = 'IAT_figures'
data_path = 'IAT_data'

############################################################
# load data
iat_data = pandas.read_csv(join(data_path,'cbsa_iat.csv'))
with open(join(data_path,'cbsa_iat_split_halves.pkl'),'rb') as f:
    split_halves = pickle.load(f)
heat_data = pandas.read_csv(join(data_path,'North America Land Data Assimilation System (NLDAS) Daily Air Temperatures and Heat Index (1979-2011).txt')
                             , sep='\t')
adi_data = pandas.read_csv(join(data_path,'US_2019_ADI_Census Block Group_v3.1.txt'))
cbsas_delineation = pandas.read_csv(join(data_path,'delineation_2020.csv'),skiprows=0) #skipsrows2
############################################################
# Prep various datasets
heat_data = heat_data[heat_data['Notes']=='Total']
heat_data = heat_data[~numpy.isnan(heat_data['County Code'])]
heat_data['county'] = heat_data['County Code'].astype(int).astype(str)
heat_data['county'] = heat_data['county'].map(lambda x: x if len(x)==5 else '0'+x)
heat_data = heat_data[['county', 'Avg Daily Max Heat Index (C)']]
heat_data = heat_data[heat_data['Avg Daily Max Heat Index (C)']!='Missing']
heat_data['Avg Daily Max Heat Index (C)'] = heat_data['Avg Daily Max Heat Index (C)'].astype(float)

adi_data['county'] = adi_data['FIPS'].astype(str).map(lambda x: x[-4:]).astype(int)

cbsas_delineation['FIPS State Code'] = cbsas_delineation['FIPS State Code'].astype(str).map(lambda x: x if len(x) == 2 else '0' + x)
cbsas_delineation['FIPS County Code'] = cbsas_delineation['FIPS County Code'].astype(str).map(
    lambda x: x if len(x) == 3 else ('0' + x) if len(x) == 2 else '00' + x)
cbsas_delineation['county'] = cbsas_delineation['FIPS State Code'].astype(str) + cbsas_delineation['FIPS County Code'].astype(str)

joined_heat = heat_data.join(cbsas_delineation,rsuffix='_').groupby('CBSA Code').mean().reset_index()[
    ['CBSA Code', 'Avg Daily Max Heat Index (C)']
]
joined_adi = adi_data.set_index('county').join(cbsas_delineation[['CBSA Code','county']])
joined_adi = joined_adi[(joined_adi['ADI_NATRANK']!='GQ') &
                        (joined_adi['ADI_NATRANK']!='GQ-PH') &
                        (joined_adi['ADI_NATRANK']!='NONE') &
                        (joined_adi['ADI_NATRANK']!='PH') &
                        (joined_adi['ADI_NATRANK']!='QDI') ]
joined_adi['ADI_NATRANK'] = joined_adi['ADI_NATRANK'].astype(float)
joined_adi = joined_adi.groupby('CBSA Code').mean().reset_index()[['CBSA Code', 'ADI_NATRANK']]

############################################################

all_r2s = []
suffixs = ['','_seg_idx','_gini','_exp']
for suffix in suffixs:
    pops = numpy.load(join(data_path,'pops'+suffix+'.npy'),allow_pickle=True)
    cbsas = numpy.load(join(data_path,'cbsas'+suffix+'.npy'),allow_pickle=True)
    white_pop = numpy.load(join(data_path,'white_pop'+suffix+'.npy'),allow_pickle=True)
    black_pop = numpy.load(join(data_path,'black_pop'+suffix+'.npy'),allow_pickle=True)
    white_hom = numpy.load(join(data_path,'white_hom'+suffix+'.npy'),allow_pickle=True)
    black_hom = numpy.load(join(data_path,'black_hom'+suffix+'.npy'),allow_pickle=True)


    plt.clf()
    plt.subplot(1,3,1)
    plt.hist(np.hstack(white_pop[-1]),alpha=0.75)
    plt.xlabel('% white')
    plt.subplot(1,3,2)
    plt.hist(np.hstack(white_hom[-1]),alpha=0.75)
    plt.xlabel('white segregation')
    plt.subplot(1,3,3)
    plt.hist(np.hstack(black_hom[-1]),alpha=0.75)
    plt.xlabel('black segregation')
    plt.tight_layout()
    plt.savefig(join(figure_root_path,'homs_and_pops_2020'+suffix+'.png'),dpi=300)

    years = list(range(2010,2021))

    all_fits = []
    thresholds = [500,250,1000]
    iat_sample_sizes = []
    heat_fits = []
    adi_fits = []
    noises = []
    for threshold in thresholds:
        print(threshold)
        iat = []
        maj_adj = []
        pop = []
        het_adj = []
        tract_fits = []
        tract_fits_white = []
        scaling_fits = []
        overall_fits = []
        slave_fits = []
        # hfs = []
        # adfs = []
        ns = []
        pop_fraction = []
        xs = []
        ys = []
        hs = []
        ws = []
        noise_r2 = []
        for i in range(len(years)):
            year_data = iat_data[(iat_data['cbsa_code'].isin(cbsas[i])) & (iat_data['year']==years[i])]
            ns.append(year_data['iat_n'].sum())
            y=year_data['iat_mean']
            iat.append([y[0] if y.shape[0]>0 else numpy.nan for y in [year_data[year_data['cbsa_code']== x]['iat_mean'].values for x in numpy.unique(cbsas_delineation['CBSA Code'])]])
            het_adj.append(
                [np.log(2 - white_hom[i].values[y[0]] - black_hom[i].values[y[0]]) if y.shape[0] > 0 else numpy.nan for
                 y in [numpy.argwhere(cbsas[i].values == x).flatten() for x in
                       numpy.unique(cbsas_delineation['CBSA Code'])]])
            maj_adj.append(
                [np.log(white_pop[i].values[y[0]] - white_pop[i].values[y[0]] ** 2) if y.shape[0] > 0 else numpy.nan for
                 y in [numpy.argwhere(cbsas[i].values == x).flatten() for x in
                       numpy.unique(cbsas_delineation['CBSA Code'])]])
            pop.append([np.log(pops[i].values[y[0]]) if y.shape[0] > 0 else numpy.nan for y in
                        [numpy.argwhere(cbsas[i].values == x).flatten() for x in
                         numpy.unique(cbsas_delineation['CBSA Code'])]])
            x=np.hstack([pops[i][cbsas[i]==x] for x in year_data['cbsa_code']])
            pop_fraction.append(numpy.median((year_data['iat_n'].values/x)[(year_data['iat_n'].values>threshold)]))            
            w=np.hstack([white_pop[i][cbsas[i]==x] for x in year_data['cbsa_code']])
            b = np.hstack([black_pop[i][cbsas[i] == x] for x in year_data['cbsa_code']])
            wh = np.hstack([white_hom[i][cbsas[i] == x] for x in year_data['cbsa_code']])
            bh = np.hstack([black_hom[i][cbsas[i] == x] for x in year_data['cbsa_code']])
            n =year_data['iat_n']
            heat = numpy.array([y.values[0][1] if y.shape[0] > 0 else numpy.nan for y in
                          [joined_heat[joined_heat['CBSA Code'].astype(int) == x] for x in year_data['cbsa_code']]])
            adi = numpy.array([y.values[0][1] if y.shape[0] > 0 else numpy.nan for y in
                                [joined_adi[joined_adi['CBSA Code'].astype(int) == x] for x in
                                 year_data['cbsa_code']]])
            split_cbsas = numpy.array([x[0] for x in split_halves[i]])
            splits = numpy.array([x[1] for x in split_halves[i]])[np.hstack([np.argwhere(split_cbsas == x).flatten() for x in year_data['cbsa_code']])]
            keep = ~np.isnan(y) & (n>threshold)
            x = x[keep]            
            w = w[keep]
            b = b[keep]
            wh = wh[keep]
            bh = bh[keep]
            splits = splits[keep]
            heat = heat[keep]
            adi = adi[keep]
            y = y[keep].values
            xs.append(x/x.mean())
            ys.append(y / y.mean())
            hs.append(((wh+bh)) / ((wh+bh)).mean())
            ws.append((w-w**2) / (w-w**2).mean())
            scaling_fit = OLS(np.log(y),add_constant(np.vstack([np.log(x)]).T)).fit()
            tract_fit = OLS(scaling_fit.resid,add_constant(np.vstack([(wh+bh),np.log(w-w**2)]).T)).fit()
            tract_fit_white = OLS(scaling_fit.resid, add_constant(np.vstack([np.log(w-w**2)]).T)).fit()
            overall_fit = OLS(np.log(y),add_constant(scaling_fit.fittedvalues
                                                     +tract_fit.fittedvalues)).fit()
            slave_fits.append(OLS(np.log(y[~np.isnan(year_data['pslave_1860'])[keep]]), add_constant(np.vstack([scaling_fit.fittedvalues[~np.isnan(year_data['pslave_1860'])[keep]]
                                + tract_fit.fittedvalues[~np.isnan(year_data['pslave_1860'])[keep]],year_data['pslave_1860'][~np.isnan(year_data['pslave_1860']) & keep]]).T)).fit().summary())
            tract_fits.append(tract_fit)
            tract_fits_white.append(tract_fit_white)
            scaling_fits.append(scaling_fit)
            overall_fits.append(overall_fit)

            noise = []
            for sp in range(500):
                order = numpy.random.permutation(range(len(y)))
                half = len(y)//2
                split1 = np.dstack(splits)[sp,0,:]
                split2 = np.dstack(splits)[sp,1,:]
                het1 = (wh+bh)[order][:half]
                het2 = (wh + bh)[order][half:]
                maj1 = (np.log(w-w**2))[order][:half]
                maj2 = np.log(w - w ** 2)[order][half:]
                p_up = (pearsonr(split1, y)[0] + pearsonr(split2, y)[0]) / 2
                p_down = pearsonr(split1, split2)[0]
                noise.append([p_up,p_down])

            noise_r2.append([numpy.vstack(noise).mean(0)[1]**2,numpy.vstack(noise).mean(0)[0]**2, overall_fit.rsquared, overall_fit.rsquared/numpy.vstack(noise).mean(0)[1]**2])

            adi_keep = ~numpy.isnan(adi)
            x = x[adi_keep]    
            w = w[adi_keep]
            b = b[adi_keep]
            wh = wh[adi_keep]
            bh = bh[adi_keep]
            adi = adi[adi_keep]
            heat = heat[adi_keep]
            y = y[adi_keep]
            scaling_fit_adi = OLS(np.log(y), add_constant(np.vstack([np.log(x)]).T)).fit()
            tract_fit_adi = OLS(scaling_fit_adi.resid,
                            add_constant(np.vstack([(wh + bh), np.log(w - w ** 2),adi]).T)).fit()
            tract_fit_adi_no_adi = OLS(scaling_fit_adi.resid,
                                add_constant(np.vstack([(wh + bh), np.log(w - w ** 2)]).T)).fit()
            overall_fit_adi = OLS(np.log(y), add_constant(scaling_fit_adi.fittedvalues
                                                      + tract_fit_adi.fittedvalues)).fit()
            overall_fit_adi_no_adi = OLS(np.log(y), add_constant(scaling_fit_adi.fittedvalues
                                                          + tract_fit_adi_no_adi.fittedvalues)).fit()
            adfs.append([overall_fit_adi,overall_fit_adi_no_adi])
            
            heat_keep = ~numpy.isnan(heat)
            x = x[heat_keep]            
            w = w[heat_keep]
            b = b[heat_keep]
            wh = wh[heat_keep]
            bh = bh[heat_keep]
            heat = heat[heat_keep]
            y = y[heat_keep]
            scaling_fit_heat = OLS(np.log(y), add_constant(np.vstack([np.log(x)]).T)).fit()
            tract_fit_heat = OLS(scaling_fit_heat.resid,
                                 add_constant(np.vstack([(wh + bh), np.log(w - w ** 2), heat]).T)).fit()
            tract_fit_heat = OLS(scaling_fit_heat.resid,
                                 add_constant(np.vstack([(wh + bh), np.log(w - w ** 2), heat]).T)).fit()
            tract_fit_heat_no_heat = OLS(scaling_fit_heat.resid,
                                         add_constant(np.vstack([-(wh + bh), np.log(w - w ** 2)]).T)).fit()
            overall_fit_heat = OLS(np.log(y), add_constant(scaling_fit_heat.fittedvalues
                                                           + tract_fit_heat.fittedvalues)).fit()
            overall_fit_heat_no_heat = OLS(np.log(y), add_constant(scaling_fit_heat.fittedvalues
                                                                   + tract_fit_heat_no_heat.fittedvalues)).fit()
            hfs.append([overall_fit_heat, overall_fit_heat_no_heat])
        noises.append(noise_r2)
        scaling_fits.append(OLS(np.log(np.hstack(ys)),add_constant(np.log(np.hstack(xs)))).fit())
        tract_fits.append(OLS(scaling_fits[-1].resid,add_constant(np.vstack([np.hstack(hs),np.log(np.hstack(ws))]).T)).fit())
        tract_fits_white.append(
            OLS(scaling_fits[-1].resid, add_constant(np.vstack([np.log(np.hstack(ws))]).T)).fit())
        overall_fits.append(OLS(np.log(np.hstack(ys)),add_constant(scaling_fits[-1].fittedvalues+tract_fits[-1].fittedvalues)).fit())
        heat_fits.append(hfs)
        adi_fits.append(adfs)
        all_fits.append([scaling_fits,tract_fits,tract_fits_white,overall_fits,slave_fits])
        iat_sample_sizes.append(ns)
        keep_cities = ~np.isnan(np.vstack(iat).sum(0)) & ~np.isnan(np.vstack(pop).sum(0)) & ~np.isnan(
            np.vstack(maj_adj).sum(0)) & ~np.isnan(np.vstack(het_adj).sum(0))
        year_fits = [OLS(np.log(np.vstack(iat)[:, keep_cities][:, k]), add_constant(np.vstack(
            [ np.vstack(maj_adj)[:, keep_cities][:, k],
             np.vstack(het_adj)[:, keep_cities][:, k]]).T)).fit() for k in range(keep_cities.sum())]
        keep_fits = np.array([~np.isnan(x.f_pvalue) for x in year_fits])
        is_long = multipletests([x.f_pvalue for x in np.array(year_fits)[keep_fits]], method='fdr_bh',alpha=.01)[0]
        long_params = np.vstack([x.params * (x.pvalues < .05) for x in np.array(year_fits)[keep_fits][is_long]])[:, 1:]


    for t in range(len(thresholds)):
        with open(join(figure_root_path,'noise_ceiling_' + suffix + '_%d' %
                  thresholds[t] + '.txt'), 'w') as f:
            if thresholds[t]==500 and suffix=='':
                f.write(pandas.DataFrame(
                    [[years[y], "[%.2f, %.2f]" % (noises[t][y][0], noises[t][y][1]), "%.2f" % noises[t][y][2], "%.2f" % noises[t][y][3]] for y in
                     range(len(years))], columns=["year", "noise ceiling $R^2$ range", "Full Sample $R^2$", "Lower Bound Noise Corrected $R^2$",]).to_latex(
                    index=False, escape=False, label='tab:noise' + suffix + str(thresholds[t]),
                    caption='Comparison of noise ceiling estimates and full sample $R^2$ for %s and a threshold of $>$%d responses per city.' % (
                    suffix, thresholds[t])))
            else:
                f.write(pandas.DataFrame(
                    [[years[y], "%.2f" % noises[t][y][2],
                      "%.2f" % noises[t][y][3]] for y in
                     range(len(years))], columns=["year", "Full Sample $R^2$",
                                                  "Lower Bound Noise Corrected $R^2$", ]).to_latex(
                    index=False, escape=False, label='tab:noise' + suffix + str(thresholds[t]),
                    caption='Comparison of noise ceiling estimates and full sample $R^2$ for %s and a threshold of $>$%d responses per city.' % (
                        suffix, thresholds[t])))

    summary_dfs = [
        pandas.DataFrame([[years[i] if i+1 <=len(years) else 'all years',
            '[%.3f,%.3f]' % (all_fits[j][0][i].conf_int()[1,0],all_fits[j][0][i].conf_int()[1,1]),
            '%.3f' % all_fits[j][0][i].rsquared_adj,
            '[%.3f,%.3f]' % (all_fits[j][1][i].conf_int()[2,0],all_fits[j][1][i].conf_int()[2,1]),
            '[%.3f,%.3f]' % (all_fits[j][1][i].conf_int()[1,0],all_fits[j][1][i].conf_int()[1,1]),
            '%.3f' % all_fits[j][2][i].rsquared_adj,
            '%.3f' % (all_fits[j][1][i].rsquared_adj-all_fits[j][2][i].rsquared_adj),
            '%.3f' % all_fits[j][3][i].rsquared_adj,
            '%d' % all_fits[j][3][i].nobs
        ] for i in range(len(years)+1)],columns=[
            'year',
            r'scaling $\beta$',
            r'scaling $R^2$',
            r'diversity $\beta$',
            r'segregation $\beta$',
            r'diversity $R^2$',
            r'segregation $R^2$',
            r'overall $R^2$',
            r'\# cities']) for j in range(len(thresholds))
                   ]
    heat_df = pandas.DataFrame([(years[i],'%.3f' % adi_fits[0][i][1].rsquared,'%.3f' % adi_fits[0][i][0].rsquared,'%d' % adi_fits[0][i][0].nobs,'%.3f' % heat_fits[0][i][1].rsquared,'%.3f' % heat_fits[0][i][0].rsquared,'%d' % heat_fits[0][i][0].nobs) for i in range(len(years))], columns=['year',
            r'no ADI $R^2$',
            r'ADI $R^2$',
            r'ADI n',
            r'no HI $R^2$',
            r'HI $R^2$',
            r'HI n',
            ])
    with open(join(figure_root_path,'heat_effect_sizes' + suffix + '.txt'), 'w') as f:
        f.write(heat_df
                .to_latex(index=False,
                          caption='Comparison of Models for cities that have available Area Deprivation Index (ADI) and Heat Index (HI) data. All models include city size, diversity and segregation effects (%s).' % suffix,
                          label='tab:heat%s' % suffix, escape=False))
    
    for i in range(len(years)):
        with open(join(figure_root_path,'sample_size_IAT'+suffix+'.txt'),'w') as f:
            f.write(pandas.DataFrame(np.vstack([years,iat_sample_sizes[0],
                                                ["%.2f%%" % x for x in np.array(pop_fraction)*100]]).T,
                                     columns=['year','IAT sample size','median CBSA population \%'])
                    .to_latex(index=False,
                              caption='IAT participants with geographic information',
                              label='tab:sampleSize',escape=False))

    for i in range(len(thresholds)):
        with open((join(figure_root_path,'summary_table_threshold_%d_params'+suffix+'.txt') % thresholds[i]), 'w') as f:
            f.write(summary_dfs[i][[
                'year',
                r'scaling $\beta$',
                r'diversity $\beta$',
                r'segregation $\beta$',
                r'\# cities'
            ]].to_latex(index=False,
                                            label='tab:summary%d' % thresholds[i],
                                            caption='Summary of scaling fits and diversity and segregation parameters for cities with more than %d IAT responses' % thresholds[i],
                                            position='hbpt!',
                                            escape=False,
                                            ))

    for i in range(len(thresholds)):
        with open((join(figure_root_path,'summary_table_threshold_%d_variance'+suffix+'.txt') % thresholds[i]), 'w') as f:
            f.write(summary_dfs[i][[
                'year',
                r'scaling $R^2$',
                r'diversity $R^2$',
                r'segregation $R^2$',
                r'overall $R^2$',
                r'\# cities'
            ]].to_latex(index=False,
                                            label='tab:summary%d' % thresholds[i],
                                            caption='Summary of scaling fits and diversity and segregation variance explained for cities with more than %d IAT responses' % thresholds[i],
                                            position='hbpt!',
                                            escape=False,
                                            ))
    plt.clf()
    rs_df = summary_dfs[0][summary_dfs[0]['year'].isin(years)]
    r2 = [rs_df[rs_df['year'] > 2015]['segregation $R^2$'].astype(float),
          rs_df[rs_df['year'] > 2015]['scaling $R^2$'].astype(float),
          rs_df[rs_df['year'] > 2015]['diversity $R^2$'].astype(float),
          rs_df[rs_df['year'] > 2015]['overall $R^2$'].astype(float)]
    all_r2s.append(r2)
    vp = plt.violinplot(r2, showextrema=False, showmedians=True)
    vp['bodies'][0].set_color(COLORS[1])
    vp['bodies'][2].set_color(COLORS[1])
    vp['bodies'][1].set_color('k')
    vp['bodies'][1].set_alpha(.75)
    vp['bodies'][3].set_color(COLORS[0])
    vp['bodies'][3].set_alpha(.75)
    plt.xticks([1, 2, 3, 4],labels=['segregation','scaling','diversity', 'overall'],rotation=0,size=13)
    plt.ylabel(r'$R^2$',size=16)
    plt.yticks(size=13)
    plt.tight_layout()
    plt.savefig(join(figure_root_path,'rsquared_violin'+suffix+'.png'),dpi=300)

    print(numpy.median(r2,1))    
    plt.clf()
    x=all_fits[0][0][-2].model.exog[:,1]
    y=all_fits[0][0][-2].model.endog
    plt.scatter(x,y,alpha=.6,label='IAT data',color=COLORS[2])
    y2 = all_fits[0][0][-2].fittedvalues+all_fits[0][1][-2].fittedvalues
    plt.scatter(x,y2,
                alpha=.75,label='segregation & diversity',color=COLORS[0])
    plt.plot(np.linspace(x.min(),x.max()),all_fits[0][0][-2].predict(
        add_constant(np.linspace(x.min(),x.max()))),color='k',linestyle='-',alpha=0.75,label='scaling')
    ord = numpy.argsort(x)
    plt.fill_between(x[ord],
                     all_fits[0][0][-2].fittedvalues[ord]-
                     all_fits[0][0][-2].get_prediction().se_mean[ord],
                     all_fits[0][0][-2].fittedvalues[ord]+
                     all_fits[0][0][-2].get_prediction().se_mean[ord],color='k',alpha=.25)
    plt.axhline(all_fits[0][0][-2].fittedvalues.max(),xmin=.05,xmax=.95,color='k',alpha=.25,linestyle='--')
    plt.ylabel('ln(bias)')
    plt.xlabel('ln(N)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(figure_root_path,'scaling_plot'+suffix+'.png'),dpi=300)

    if suffix=='':
        fig2adf = pandas.DataFrame({'city pop':x,'city mean iat':y,'city residuals': y2}).to_csv(join(figure_root_path,'fig2a.csv'))
        a_mean = numpy.mean([x.params[1] / (1 / 6) for x in all_fits[0][0]])
        a_min = numpy.mean([x.conf_int()[1, 0] / (1 / 6) for x in all_fits[0][0]])
        a_max = numpy.mean([x.conf_int()[1, 1] / (1 / 6) for x in all_fits[0][0]])
        a_max_diversity = numpy.mean([x.conf_int()[2, 0] for x in all_fits[0][1]])
        a_min_diversity = numpy.mean([x.conf_int()[2, 1] for x in all_fits[0][1]])
        a_mean_diversity = numpy.mean([x.params[2] for x in all_fits[0][1]])
        pandas.DataFrame({'learning rate estimate': [a_mean,
                                                     a_min,a_max,
                                                     a_mean_diversity,
                                                     a_min_diversity,
                                                     a_max_diversity],
                          'source':[
                              'scaling mean',
                              'scaling confidence interval lower bound',
                              'scaling confidence interval upper bound',
                              'diversity mean',
                              'diversity confidence interval lower bound',
                              'diversity confidence interval upper bound',
                          ]}).to_csv(join(
            figure_root_path, 'figure3estimates.csv'
        ))
        experimental_interventions = numpy.array([
            0.35 / 0.45,
            0.22 / 0.43,
            0.24 / 0.43,
            0.22 / 0.43,
            0.17 / 0.50,
            0.32 / 0.50,
            0.26 / 0.50,
            0.19 / 0.50,
            0.16 / 0.42,
            0.25 / 0.42,
            0.24 / 0.42,
            0.32 / 0.42,  # all of these from Lai et al 2014
            0.25 / 0.61,
            0.36 / 0.63,
            0.49 / 0.64,
            0.27 / 0.58,
            0.32 / 0.59,
            0.30 / 0.56,
            0.34 / 0.57,  # all of these from lai et all 2020
        ])
        pandas.DataFrame({'experimental effects': experimental_interventions}).to_csv(join(
            figure_root_path,'figure3experimental.csv'
        ))
        plt.clf()
        x = np.linspace(0, 30)
        plt.plot(x, (x + 1) ** (a_mean), color=COLORS[0], label='estimated from scaling exponent')
        plt.plot(x, (x + 1) ** (a_mean_diversity), color=COLORS[1], label='estimated from diversity effects')
        plt.fill_between(x, (x + 1) ** (a_min_diversity), (x + 1) ** (a_max_diversity), alpha=.25, color=COLORS[1])
        plt.fill_between(x, (x + 1) ** (a_min), (x + 1) ** (a_max), alpha=.25, color=COLORS[0])
        vp = plt.violinplot(experimental_interventions, positions=[2], widths=[2], showextrema=True, showmeans=True)
        vp['bodies'][0].set_color(COLORS[-1])
        vp['cmeans'].set_color(COLORS[-1])
        vp['cmaxes'].set_color(COLORS[-1])
        vp['cmins'].set_color(COLORS[-1])
        vp['cbars'].set_color(COLORS[-1])
        plt.text(-.5, .28, r'        Upper Bound on $\alpha$' + '\nfrom experimental inventions')
        plt.text(14, .325, r'$\alpha=$%.2f' % -a_min, rotation=-20)
        plt.text(16.2, .755, r'$\alpha=$%.2f' % -a_min_diversity, rotation=-5)
        plt.text(15, .485, r'$\alpha_{scaling}=$%.2f' % -a_mean, rotation=-12)
        plt.text(15.5, .575, r'$\alpha_{diversity}=$%.2f' % -a_mean_diversity, rotation=-9)
        plt.yscale('log')
        plt.xlabel(r'Additional Inter-Group Contacts (#)')
        plt.ylabel(r'$b/b_0$')
        plt.yticks([1, .9, .8, .7, .6, .5, .4, .3], labels=['100%', '90%', '80%', '70%', '60%', '50%', '40%', '30%'])
        plt.tight_layout()
        plt.savefig(join(figure_root_path,'learning_curve_' + suffix + '.png'), dpi=300)

print('all R2s for different segregation measures')
print(numpy.median(numpy.vstack([numpy.vstack(x).T for x in all_r2s]),0))
print(numpy.max(numpy.vstack([numpy.vstack(x).T for x in all_r2s]),0))
print(numpy.min(numpy.vstack([numpy.vstack(x).T for x in all_r2s]),0))
r2 = numpy.vstack([numpy.vstack(x).T for x in all_r2s])
plt.clf()
plt.axhline(numpy.median(numpy.vstack(noises),0)[0],linestyle='--',alpha=.25)
plt.axhline(numpy.median(numpy.vstack(noises)/2,0)[0],linestyle='--',alpha=.75)
vp = plt.violinplot(r2, showextrema=False, showmedians=True)
vp['bodies'][0].set_color(COLORS[1])
# vp['cmeans'][0].set_color(COLORS[1])
vp['bodies'][2].set_color(COLORS[1])
# vp['cmeans'][2].set_color(COLORS[1])
vp['bodies'][1].set_color('k')
# vp['cmeans'][1].set_color('k')
vp['bodies'][1].set_alpha(.75)
vp['bodies'][3].set_color(COLORS[0])
# vp['cmeans'][3].set_color(COLORS[0])
vp['bodies'][3].set_alpha(.75)
vp['bodies'][3].set_zorder(2)
plt.xticks([1, 2, 3, 4], labels=['segregation', 'scaling', 'diversity', 'overall'],
           rotation=0, size=13)
plt.ylabel(r'$R^2$', size=16)
plt.yticks(size=13)
plt.ylim(0,.65)
plt.text(.6,.55,'median noise ceiling lower bound',alpha=.5)
plt.text(.6,.28,'50% of median noise ceiling')
plt.tight_layout()
plt.savefig(join(figure_root_path,'rsquared_violin_all.png'), dpi=300)

pandas.DataFrame({'segregation r2':r2[:,0],'scaling r2':r2[:,1],'diversity r2':r2[:,2],'overall r2':r2[:,3]}).to_csv(
    join(figure_root_path,'figure2b.csv')
)

pandas.DataFrame({'noise ceiling':numpy.vstack(noises).flatten()}).to_csv(
    join(figure_root_path,'figure2b_noise.csv')
)


years = range(2010,2021)
for year in years:
    data = pandas.read_csv(join(data_path,'cbsa_iat_individual_%d.csv' % year))
    data = data[data['edu_14']!=' ']
    data = data[data['edu_14'].astype(str) != 'nan']
    if np.unique(data['birthsex'].astype(str)).shape[0]>2:
        data = data[data['birthsex'] != ' ']
        data = data[data['birthsex'].astype(str) != 'nan']
    if year>=2016:
        data = data[data['raceomb_002'].astype(str)!=' ']
        data = data[data['raceomb_002'].astype(str) != 'nan']
    else:
        data = data[data['raceomb'].astype(str) != ' ']
        data = data[data['raceomb'].astype(str) != 'nan']
    data = data[~np.isnan(data['cbsa_pop'])]
    data['bias'] = data['D_biep.White_Good_all']>0
    data['D_biep.White_Good_all'] = data['D_biep.White_Good_all'] + np.abs(data['D_biep.White_Good_all'].min()) + 1
    bsdf = data[~numpy.isnan(data['birthsex'].astype(float))]
    if bsdf.shape[0]>0:
        print((bsdf['birthsex']=='1').sum(),(bsdf['birthsex']=='2').sum(),(bsdf['birthsex']=='1').sum()+(bsdf['birthsex']=='2').sum(),len(bsdf))
    df = pandas.DataFrame(np.vstack([
        np.log(data['cbsa_pop'].values),
        categorical(data['raceomb_002' if year >2016 else 'raceomb'].values)[:, [6, 5, 8]].astype(int).T,
        categorical(data['birthsex'].values)[:, [1]].astype(int).T,
        categorical(data['edu_14'].values)[:, [1, 2, 3, 4]].sum(1).astype(int),
        categorical(data['edu_14'].values)[:, [5, 6, 7]].sum(1).astype(int),
        categorical(data['edu_14'].values)[:, 8:].sum(1).astype(int),
        data['maj_group_adjust'].values,
        data['het_correction'].values
    ]).T, columns=[
        'ln(population)',
        'White',
        'Black',
        'Multiracial',
        'Birth Sex',
        'High School or Less',
        'College',
        'Advanced Degree',
        'Diversity',
        'Segregation'
    ])
    means = df.mean()
    cols_to_keep = np.array(list(df.columns))[means!=0]

    df = df[cols_to_keep]
    df = df.reset_index().set_index('index')
    individual_fit = Logit(data.reset_index()['bias'], add_constant(df)).fit()
    if np.isnan(individual_fit.pvalues.values[0]):
        individual_fit = Logit(data.reset_index()['bias'], add_constant(df.drop('High School or Less',inplace=False,axis=1))).fit()
    with open(join(figure_root_path,'individual_iat_fit_%d.txt' % year), 'w') as f:
        f.write(individual_fit.summary().as_latex())

print()

