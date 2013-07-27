import pymc
import numpy as np
import pylab as pl
from measure_tau import mcmc_sampler_dict,tauoneone,tautwotwo,savepath,domillion,abundance,savefig,trace_data_path
from mcmc_tools import docontours_multi,save_traces
from agpy import pymc_plotting
import pymc_tools

print "Beginning Lognormal parameter estimation using abundance=",abundance

mc_simple = pymc.MCMC(mcmc_sampler_dict(tauoneone=tauoneone,tautwotwo=tautwotwo))

graph_lognormal_simple = pymc.graph.graph(mc_simple)
graph_lognormal_simple.write_pdf(savepath+"mc_lognormal_simple_graph.pdf")
graph_lognormal_simple.write_png(savepath+"mc_lognormal_simple_graph.png")

d = mcmc_sampler_dict(tauoneone=tauoneone,tautwotwo=tautwotwo,truncate_at_50sigma=True)
d['b'] = pymc.Uniform(name='b', value=0.5, lower=0.3, upper=1, observed=False)
@pymc.deterministic(plot=True,trace=True)
def mach(sigma=d['sigma'], b=d['b']):
    return np.sqrt((np.exp(sigma**2) - 1)/b**2)

d['mach'] = mach
d['mach_observed'] = pymc.Normal(name='mach_observed', mu=mach, tau=1./1.5**2, value=5.1, observed=True)

mc_lognormal = pymc.MCMC(d)

d = mcmc_sampler_dict(tauoneone=tauoneone,tautwotwo=tautwotwo,truncate_at_50sigma=True)
d['b'] = pymc.Uniform(name='b', value=0.5, lower=0.3, upper=1, observed=False)
@pymc.deterministic(plot=True,trace=True)
def mach(sigma=d['sigma'], b=d['b']):
    return np.sqrt((np.exp(sigma**2) - 1)/b**2)

d['mach'] = mach
def mach_limits(m):
    if m < 1 or m > 50:
        return -np.inf
    else:
        return 0
d['sigma'].value = 1.00 # abund -9
d['sigma'].value = 1.75 # abund -8.5
d['mach_limits'] = pymc.Potential(name='mach_observed', logp=mach_limits, parents={'m':d['mach']},doc='Mach limits',verbose=0)

mc_lognormal_freemach = pymc.MCMC(d)

graph_lognormal = pymc.graph.graph(mc_lognormal)
graph_lognormal.write_pdf(savepath+"mc_lognormal_graph.pdf")
graph_lognormal.write_png(savepath+"mc_lognormal_graph.png")


print "\nSimple sampling (initial)"
mc_simple.sample(100)
print "\nlognormal sampling (initial)"
mc_lognormal.sample(100)
print "\nlognormal (freemach) sampling (initial)"
mc_lognormal_freemach.sample(100)

print "\nSimple sampling"
mc_simple.sample(100000)
print "\nlognormal sampling"
mc_lognormal.sample(100000)
print "\nlognormal (freemach) sampling"
mc_lognormal_freemach.sample(100000)

def docontours_all(mc_lognormal,mc_simple,mc_lognormal_freemach):
    docontours_multi(mc_lognormal,start=10000,savename=savepath+"mc_lognormal_withmach_multipanel_abundance%s.pdf" % abundance, dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio'))
    #docontours_multi(mc_lognormal,start=10000,savename=savepath+"mc_lognormal_withmach_multipanel_deviance.pdf", dosave=True,
    #                 parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio','deviance'))
    #docontours_multi(mc_lognormal,start=10000,savename=savepath+"mc_lognormal_withmach_multipanel_scalefactors.pdf", dosave=True,
    #                 parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio','Metropolis_sigma_adaptive_scale_factor','Metropolis_b_adaptive_scale_factor','Metropolis_meandens_adaptive_scale_factor','deviance'))
    #docontours_multi(mc_lognormal,start=10000,savename=savepath+"mc_lognormal_withmach_multipanel_deviance.pdf", dosave=True,
    #                 parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio','deviance'))
    docontours_multi(mc_simple,start=10000,savename=savepath+"mc_lognormal_justtau_multipanel_abundance%s.pdf" % abundance, dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','tau_ratio'))

    docontours_multi(mc_lognormal_freemach,start=10000,savename=savepath+"mc_lognormal_freemach_multipanel_abundance%s.pdf" % abundance, dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio'))
    #docontours_multi(mc_lognormal_freemach,start=10000,savename=savepath+"mc_lognormal_freemach_multipanel_deviance.pdf", dosave=True,
    #                 parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio','deviance'))
    #docontours_multi(mc_lognormal_freemach,start=10000,savename=savepath+"mc_lognormal_freemach_multipanel_scalefactors.pdf", dosave=True,
    #                 parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio','Metropolis_sigma_adaptive_scale_factor','Metropolis_b_adaptive_scale_factor','Metropolis_meandens_adaptive_scale_factor','deviance'))
#varslice=(10000,None,None)
#for fignum,(p1,p2) in enumerate(itertools.combinations(('tauoneone_mu','tautwotwo_mu','tau_ratio','sigma','meandens','b','mach'),2)):
#    pymc_plotting.hist2d(mc_lognormal, p1, p2, bins=30, clear=True, fignum=fignum, varslice=varslice, colorbar=True)
#    pl.title("Lognormal with Mach")
#    pl.savefig(savepath+"LognormalWithMach_%s_v_%s_mcmc.png" % (p1,p2))

#for fignum2,(p1,p2) in enumerate(itertools.combinations(('tauoneone_mu','tautwotwo_mu','tau_ratio','sigma','meandens'),2)):
#    pymc_plotting.hist2d(mc_simple, p1, p2, bins=30, clear=True, fignum=fignum+fignum2+1, varslice=varslice, colorbar=True)
#    pl.title("Lognormal - just $\\tau$ fits")
#    pl.savefig(savepath+"LognormalJustTau_%s_v_%s_mcmc.png" % (p1,p2))

docontours_all(mc_lognormal,mc_simple,mc_lognormal_freemach)

if domillion:
    print "\nSimple sampling 1 million"
    mc_simple.sample(1e6)
    mc_lognormal_simple_traces = save_traces(mc_simple, trace_data_path+"mc_lognormal_simple_traces%s.fits" % abundance, clobber=True)
    print "\nlognormal sampling 1 million"
    mc_lognormal.sample(1e6)
    mc_lognormal_traces = save_traces(mc_lognormal, trace_data_path+"mc_lognormal_withmach_traces%s.fits" % abundance, clobber=True)
    print "\nlognormal (freemach) sampling 1 million"
    mc_lognormal_freemach.sample(1e6)
    mc_lognormal_freemach_traces = save_traces(mc_lognormal_freemach, trace_data_path+"mc_lognormal_freemach_traces%s.fits" % abundance, clobber=True)

    docontours_all(mc_lognormal,mc_simple,mc_lognormal_freemach)

lognormal_statstable = pymc_tools.stats_table(mc_lognormal)
lognormal_statstable.write(trace_data_path+'lognormal_statstable_abundance%s.fits' % abundance, overwrite=True)
lognormal_simple_statstable = pymc_tools.stats_table(mc_simple)
lognormal_simple_statstable.write(trace_data_path+'lognormal_simple_statstable_abundance%s.fits' % abundance, overwrite=True)
lognormal_freemach_statstable = pymc_tools.stats_table(mc_lognormal_freemach)
lognormal_freemach_statstable.write(trace_data_path+'lognormal_freemach_statstable_abundance%s.fits' % abundance, overwrite=True)


pl.figure(33)
pl.clf()
pl.title("Lognormal")
pymc_plotting.plot_mc_hist(mc_lognormal,'b',lolim=True,alpha=0.5,bins=25,legloc='lower right')
pl.xlabel('$b$')
savefig(savepath+'LognormalWithMach_b_1D_restrictions.png')

print "Some statistics used in the paper: "
print 'mc_lognormal_simple sigma: ',mc_simple.stats()['sigma']['quantiles']
print 'mc_lognormal        sigma: ',mc_lognormal.stats()['sigma']['quantiles']
print 'mc_lognormal        b: ',mc_lognormal.stats(quantiles=(0.1,1,2.5,5,50))['b']['quantiles']
