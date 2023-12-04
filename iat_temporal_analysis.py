import glob
import pandas
import matplotlib.pyplot as plt
import numpy
from met_brewer.palettes import met_brew
from statsmodels.tsa.stattools import grangercausalitytests
from os.path import join

COLORS = met_brew('Morgenstern',n=6,brew_type="continuous")
STATE_FIPS = {"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69","PR":"72","VI":"78"}


figure_root_path = 'IAT_figures'
data_path = 'IAT_data'


iat_data = pandas.read_csv(join(data_path,'cbsa_iat.csv'))


threshold = 500
count_years = iat_data[~numpy.isnan(iat_data['iat_mean'])& (iat_data['iat_n']>threshold)].groupby(['cbsa_code']).count().reset_index()[['year','cbsa_code']]
longitudinal_cbsa = numpy.unique(count_years[count_years['year']==11]['cbsa_code'])

suffixs = ['','_seg_idx','_gini','_exp']
sps = []
spse = []
all_fits_f_tests = []
for suffix in suffixs:
    gc_fits = []
    white_pop = numpy.load(join(data_path,'white_pop'+suffix+'.npy'),allow_pickle=True)
    black_pop = numpy.load(join(data_path,'black_pop'+suffix+'.npy'),allow_pickle=True)
    white_hom = numpy.load(join(data_path,'white_hom'+suffix+'.npy'),allow_pickle=True)
    black_hom = numpy.load(join(data_path,'black_hom'+suffix+'.npy'),allow_pickle=True)
    pops = numpy.load(join(data_path,'pops' + suffix + '.npy'), allow_pickle=True)
    cbsas = numpy.load(join(data_path,'cbsas' + suffix + '.npy'), allow_pickle=True)
    years = numpy.unique(iat_data['year'])
    by_year = []
    for i in range(len(years)):
        year_data = iat_data[iat_data['year'] == years[i]]
        year_data = year_data[year_data['cbsa_code'].isin(cbsas[i].values)]
        year_data = year_data[~numpy.isnan(year_data['iat_mean'])]
        year_data = year_data[year_data['iat_n'] > threshold]
        year_data['wh'] =numpy.hstack([white_hom[i].values[numpy.argwhere(cbsas[i].values==x).flatten()] for x in year_data['cbsa_code']])
        year_data['bh'] = numpy.hstack([black_hom[i].values[numpy.argwhere(cbsas[i].values == x).flatten()] for x in year_data['cbsa_code']])
        year_data['wp'] = numpy.hstack([white_pop[i].values[numpy.argwhere(cbsas[i].values == x).flatten()] for x in year_data['cbsa_code']])
        year_data['pop'] = numpy.hstack([pops[i].values[numpy.argwhere(cbsas[i].values == x).flatten()] for x in year_data['cbsa_code']])
        by_year.append(year_data)
        print()
    by_year = pandas.concat(by_year)
    by_year = by_year.set_index(['cbsa_code','year'])
    longitudinal = by_year.reset_index()[by_year.reset_index()['cbsa_code'].isin(longitudinal_cbsa)]
    longitudinal['seg'] = longitudinal['wh'] + longitudinal['bh']
    longitudinal['div'] = numpy.log(longitudinal['wp'] - longitudinal['wp'] ** 2)
    longitudinal['iat_mean'] = numpy.log(longitudinal['iat_mean'])
    longitudinal['pop'] = numpy.log(longitudinal['pop'])
    n = len(numpy.unique(longitudinal['cbsa_code']))
    lags = [1,2,3]
    gcpop1 = []
    gcpop2 = []
    gcpop1f = []
    gcpop2f = []
    popfits = []
    for lag in lags:
        gcpopfits1 = [
            grangercausalitytests(numpy.diff([x for x in longitudinal.groupby('cbsa_code')][i][1][['iat_mean', 'pop']], 0),
                                  maxlag=[lag])[lag][0] for i in range(n)]
        gcpopfits2 = [
            grangercausalitytests(numpy.diff([x for x in longitudinal.groupby('cbsa_code')][i][1][['pop', 'iat_mean']], 0),
                                    maxlag=[lag])[lag][0] for i in range(n)]
        gcpop1.append(numpy.vstack([[y[1] for y in x.values()] for x in gcpopfits1]))
        gcpop2.append(numpy.vstack([[y[1] for y in x.values()] for x in gcpopfits2]))
        gcpop1f.append(numpy.vstack([[y[0] for y in x.values()] for x in gcpopfits1]))
        gcpop2f.append(numpy.vstack([[y[0] for y in x.values()] for x in gcpopfits2]))
        popfits.append([[y['params_ftest'] for y in gcpopfits1],[y['params_ftest'] for y in gcpopfits2]])
    gc_fits.append(['pop',popfits])
    gcseg1 = []
    gcseg2 = []
    segfits = []
    for lag in lags:
        gcseg1fits = [
            grangercausalitytests(numpy.diff([x for x in longitudinal.groupby('cbsa_code')][i][1][['iat_mean', 'seg']], 0),
                                  maxlag=[lag])[lag][0] for i in range(n)]
        gcseg2fits = [
            grangercausalitytests(numpy.diff([x for x in longitudinal.groupby('cbsa_code')][i][1][['seg', 'iat_mean']], 0),
                                    maxlag=[lag])[lag][0] for i in range(n)]
        gcseg1.append(numpy.vstack([[y[1] for y in x.values()] for x in gcseg1fits]))
        gcseg2.append(numpy.vstack([[y[1] for y in x.values()] for x in gcseg2fits]))
        segfits.append([[y['params_ftest'] for y in gcseg1fits],[y['params_ftest'] for y in gcseg2fits]])
    gc_fits.append(['seg',segfits])

    gcdiv1 = []
    gcdiv2 = []
    gcdivfits = []
    for lag in lags:
        gcdiv1fits = [
            grangercausalitytests(numpy.diff([x for x in longitudinal.groupby('cbsa_code')][i][1][['iat_mean', 'div']], 0),
                                  maxlag=[lag])[lag][0] for i in range(n)]
        gcdiv2fits = [
            grangercausalitytests(numpy.diff([x for x in longitudinal.groupby('cbsa_code')][i][1][['div', 'iat_mean']], 0),
                                    maxlag=[lag])[lag][0] for i in range(n)]
        gcdiv1.append(numpy.vstack([[y[1] for y in x.values()] for x in gcdiv1fits]))
        gcdiv2.append(numpy.vstack([[y[1] for y in x.values()] for x in gcdiv2fits]))
        gcdivfits.append([[y['params_ftest'] for y in gcdiv1fits],[y['params_ftest'] for y in gcdiv2fits]])
    gc_fits.append(['div',gcdivfits])

    x = range(1, 4)
    y1 = [(gcpop1[i] < .05)[:, 1].mean() for i in range(3)]
    y2 = [(gcpop2[i] < .05)[:, 1].mean() for i in range(3)]
    y1err = [numpy.std([numpy.random.choice((gcpop1[i] < .05)[:, 1], n, replace=True).mean() for k in range(1000)]) for
             i in range(3)]
    y2err = [numpy.std([numpy.random.choice((gcpop2[i] < .05)[:, 1], n, replace=True).mean() for k in range(1000)]) for
             i in range(3)]
    y1pop = y1
    y2pop = y2
    y1poperr = y1err
    y2poperr = y2err
    plt.clf()
    ax = plt.subplot(111)
    pps = [y1, y2]
    ppse = [y1err, y2err]
    plt.errorbar(x, y1, y1err, label='population -> bias', capsize=4, alpha=.75, color=COLORS[0])
    plt.errorbar(x, y2, y2err, label='bias -> population', capsize=4, alpha=0.75, color=COLORS[-1])
    y1 = [(gcseg1[i] < .05)[:, 1].mean() for i in range(3)]
    y2 = [(gcseg2[i] < .05)[:, 1].mean() for i in range(3)]
    plt.ylabel(r'% of Cities $p<0.05$')
    plt.xticks([1, 2, 3])
    plt.xlabel('lag (years)')
    plt.xlim(0.75, 3.25)
    plt.ylim(-.1, 1)
    ax1 = ax.inset_axes([0.65, 0.13, 0.32, 0.25])
    ax1.set_xlabel('lag (years)')
    ax1.set_xticks([1, 2, 3])
    ax1.set_ylabel(r'% of Cities $p<0.05$')
    y1err = [numpy.std([numpy.random.choice((gcseg1[i] < .05)[:, 1], n, replace=True).mean() for k in range(1000)]) for
             i in range(3)]
    y2err = [numpy.std([numpy.random.choice((gcseg2[i] < .05)[:, 1], n, replace=True).mean() for k in range(1000)]) for
             i in range(3)]
    y1seg = y1
    y2seg = y2
    y1segerr = y1err
    y2segerr = y2err
    sps.append([y1, y2])
    spse.append([y1err, y2err])
    ax1.errorbar(x, y1, y1err, capsize=4, alpha=.75, color=COLORS[0], linestyle='--')
    ax1.errorbar(x, y2, y2err, capsize=4, alpha=0.75, color=COLORS[-1], linestyle='--')
    y1 = [(gcdiv1[i] < .05)[:, 1].mean() for i in range(3)]
    y2 = [(gcdiv2[i] < .05)[:, 1].mean() for i in range(3)]
    y1err = [numpy.std([numpy.random.choice((gcdiv1[i] < .05)[:, 1], n, replace=True).mean() for k in range(1000)]) for
             i in range(3)]
    y2err = [numpy.std([numpy.random.choice((gcdiv2[i] < .05)[:, 1], n, replace=True).mean() for k in range(1000)]) for
             i in range(3)]
    y1div = y1
    y2div = y2
    y1diverr = y1err
    y2diverr = y2err
    ax1.errorbar(x, y1, y1err, capsize=4, alpha=.75, color=COLORS[0], linestyle=':')
    ax1.errorbar(x, y2, y2err, capsize=4, alpha=0.75, color=COLORS[-1], linestyle=':')
    dds = [y1, y2]
    ddse = [y1err, y2err]
    legend2 = plt.legend(['population -> bias', 'bias -> population'])
    l1, = plt.plot([], [], '--', color='k', label='diversity')
    l2, = plt.plot([], [], ':', color='k', label='segregation')
    legend1 = plt.legend(['segregation', 'diversity'], bbox_to_anchor=(0.66, 0.55))
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.savefig(join(figure_root_path,'temporal_all%s.png'%suffix), dpi=300)
    if suffix == '_gini':
        pandas.DataFrame({
            'population->bias': y1pop,
            'population->bias std': y1poperr,
            'bias->population': y2pop,
            'bias->population std': y2poperr,

            'segregation->bias': y1seg,
            'segregation->bias std': y1segerr,
            'bias->segregation': y2seg,
            'bias->segregation std': y2segerr,

            'diversity->bias': y1div,
            'diversity->bias std': y1diverr,
            'bias->diversity': y2div,
            'bias->diversity std': y2diverr,
        }).to_csv(join(figure_root_path,'fig4.csv'))

    all_fits_f_tests.append([suffix,gc_fits])

def conv(arr):
    return [(r'%.1f' % (x * 100)) for x in arr]


seg1 = conv(numpy.vstack([x[0] for x in sps]).mean(0))
seg1e = conv(numpy.sqrt((numpy.vstack([x[0] for x in spse])**2).sum(0))/numpy.sqrt(4))
seg2 = conv(numpy.vstack([x[1] for x in sps]).mean(0))
seg2e = conv(numpy.sqrt((numpy.vstack([x[1] for x in spse])**2).sum(0))/numpy.sqrt(4))
div1 = conv(dds[0])
div1e = conv(ddse[0])
div2 = conv(dds[1])
div2e = conv(ddse[1])
pop1 = conv(pps[0])
pop1e = conv(ppse[0])
pop2 = conv(pps[1])
pop2e = conv(ppse[1])

pop1 = ['$' + x[0] + "\pm "+ x[1]+'$' for x in zip(pop1,pop1e)]
pop2 = ['$' + x[0] + "\pm "+ x[1]+'$' for x in zip(pop2,pop2e)]
seg1 = ['$' + x[0] + "\pm "+ x[1]+'$' for x in zip(seg1,seg1e)]
seg2 = ['$' + x[0] + "\pm "+ x[1]+'$' for x in zip(seg2,seg2e)]
div1 = ['$' + x[0] + "\pm "+ x[1]+'$' for x in zip(div1,div1e)]
div2 = ['$' + x[0] + "\pm "+ x[1]+'$' for x in zip(div2,div2e)]

with open(join(figure_root_path,'temporal_table.txt'),'w') as f:
    f.write(
        pandas.DataFrame(numpy.vstack(([ r'population$\rightarrow$bias', r'bias$\rightarrow$population',
                                         r'diversity$\rightarrow$bias', r'bias$\rightarrow$diversity',
                                         r'segregation$\rightarrow$bias', r'bias$\rightarrow$segregation',
                                         ],
                                       numpy.vstack((pop1, pop2, seg1, seg2, div1, div2)).T)).T)
        .to_latex(escape=False, index=False, header=False)
    )

c=0
with open(join(figure_root_path,'complete_model_summaries_test.csv'),'w') as f:
    f.write('Granger Causality Models F Test Results\n')
    f.write('Threshold: at least %d participants\n' % threshold)
    for t in all_fits_f_tests:
        if t[0] == '':
            suffix = "Mean Exposure"
        if t[0] == '_seg_idx':
            suffix = "Segregation Index"
        if t[0] == '_gini':
            suffix = "Gini Coefficient"
        if t[0] == '_exp':
            suffix = "Exposure Index"
        f.write('Segregation Measure: %s\n' % suffix)
        dfs = t[1]
        for r in dfs:
            if r[0] == 'pop':
                var = 'Population'
            if r[0] == 'seg':
                var = 'Segregation'
            if r[0] == 'div':
                var = 'Diversity'

            for lag in range(len(r[1])):
                f.write('Lag: %d\n' % (lag+1))
                results = r[1][lag]
                f.write('%s -> IAT\n' % var)
                f.write('City #, F, p, df1, df2\n')
                for i in range(len(results[0])):
                    f.write('%d, %.3f, %s, %d, %d\n' % (i+1,results[0][i][0],'%.3f' % results[0][i][1] if results[0][i][1]>0.001 else '<0.001',results[0][i][2],results[0][i][3]))
                    c += 1
                f.write('\n')
                f.write('IAT -> %s\n' % var)
                f.write('City #, F, p, df1, df2\n')
                for i in range(len(results[1])):
                    f.write('%d, %.3f, %s, %d, %d\n' % (i+1,results[1][i][0],'%.3f' % results[1][i][1] if results[1][i][1]>0.001 else '<0.001',results[1][i][2],results[1][i][3]))
                    c+=1
    f.write('\n\n')

print(c)







