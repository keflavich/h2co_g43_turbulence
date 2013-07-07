import pymc
import numpy as np
import pylab as pl
from measure_tau import mcmc_sampler_dict,tauoneone,tautwotwo,savepath,domillion,abundance,savefig
from mcmc_tools import docontours_multi,save_traces
from agpy import pymc_plotting
import pymc_tools

mc_simple = pymc.MCMC(mcmc_sampler_dict(tauoneone=tauoneone,tautwotwo=tautwotwo))
print "Simple sampling\n"
mc_simple.sample(100000)

graph_lognormal_simple = pymc.graph.graph(mc_simple)
graph_lognormal_simple.write_pdf(savepath+"mc_lognormal_simple_graph.pdf")
graph_lognormal_simple.write_png(savepath+"mc_lognormal_simple_graph.png")

d = mcmc_sampler_dict(tauoneone=tauoneone,tautwotwo=tautwotwo,truncate_at_5sigma=True)
d['b'] = pymc.Uniform(name='b', value=0.5, lower=0.3, upper=1, observed=False)
@pymc.deterministic(plot=True,trace=True)
def mach(sigma=d['sigma'], b=d['b']):
    return np.sqrt((np.exp(sigma**2) - 1)/b**2)

d['mach'] = mach
d['mach_observed'] = pymc.Normal(name='mach_observed', mu=mach, tau=1./0.2**2, value=5.1, observed=True)

mc_lognormal = pymc.MCMC(d)
print "lognormal sampling\n"
mc_lognormal.sample(100000)

d = mcmc_sampler_dict(tauoneone=tauoneone,tautwotwo=tautwotwo,truncate_at_5sigma=False)
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
d['sigma'].value = 1
d['mach_limits'] = pymc.Potential(name='mach_observed', logp=mach_limits, parents={'m':d['mach']},doc='Mach limits',verbose=0)

mc_lognormal_freemach = pymc.MCMC(d)
print "lognormal (freemach) sampling\n"
mc_lognormal_freemach.sample(100000)

graph_lognormal = pymc.graph.graph(mc_lognormal)
graph_lognormal.write_pdf(savepath+"mc_lognormal_graph.pdf")
graph_lognormal.write_png(savepath+"mc_lognormal_graph.png")

def docontours_all(mc_lognormal,mc_simple,mc_lognormal_freemach):
    docontours_multi(mc_lognormal,start=10000,savename=savepath+"mc_lognormal_withmach_multipanel.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio'))
    docontours_multi(mc_lognormal,start=10000,savename=savepath+"mc_lognormal_withmach_multipanel_deviance.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio','deviance'))
    docontours_multi(mc_lognormal,start=10000,savename=savepath+"mc_lognormal_withmach_multipanel_scalefactors.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio','Metropolis_sigma_adaptive_scale_factor','Metropolis_b_adaptive_scale_factor','Metropolis_meandens_adaptive_scale_factor','deviance'))
    docontours_multi(mc_lognormal,start=10000,savename=savepath+"mc_lognormal_withmach_multipanel_deviance.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio','deviance'))
    docontours_multi(mc_simple,start=10000,savename=savepath+"mc_lognormal_justtau_multipanel.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','tau_ratio'))

    docontours_multi(mc_lognormal_freemach,start=10000,savename=savepath+"mc_lognormal_freemach_multipanel.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio'))
    docontours_multi(mc_lognormal_freemach,start=10000,savename=savepath+"mc_lognormal_freemach_multipanel_deviance.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio','deviance'))
    docontours_multi(mc_lognormal_freemach,start=10000,savename=savepath+"mc_lognormal_freemach_multipanel_scalefactors.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','mach','b','tau_ratio','Metropolis_sigma_adaptive_scale_factor','Metropolis_b_adaptive_scale_factor','Metropolis_meandens_adaptive_scale_factor','deviance'))
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
    print "Simple sampling 1 million\n"
    mc_simple.sample(1e6)
    print "lognormal sampling 1 million\n"
    mc_lognormal.sample(1e6)
    print "lognormal (freemach) sampling 1 million\n"
    mc_lognormal_freemach.sample(1e6)

    docontours_all(mc_lognormal,mc_simple,mc_lognormal_freemach)

lognormal_statstable = pymc_tools.stats_table(mc_lognormal)
lognormal_statstable.write('lognormal_statstable_abundance%s.fits' % abundance, overwrite=True)
lognormal_simple_statstable = pymc_tools.stats_table(mc_simple)
lognormal_simple_statstable.write('lognormal_simple_statstable_abundance%s.fits' % abundance, overwrite=True)
lognormal_freemach_statstable = pymc_tools.stats_table(mc_lognormal_freemach)
lognormal_freemach_statstable.write('lognormal_freemach_statstable_abundance%s.fits' % abundance, overwrite=True)

mc_lognormal_traces = save_traces(mc_lognormal, "mc_lognormal_traces", clobber=True)
mc_lognormal_simple_traces = save_traces(mc_simple, "mc_lognormal_simple_traces", clobber=True)
mc_lognormal_freemach_traces = save_traces(mc_lognormal_freemach, "mc_lognormal_freemach_traces", clobber=True)

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
