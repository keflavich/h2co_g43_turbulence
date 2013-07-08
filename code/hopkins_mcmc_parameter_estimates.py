import pymc
import numpy as np
import pylab as pl
from measure_tau import mcmc_sampler_dict,tauoneone_hopkins,tautwotwo_hopkins,savepath,domillion,abundance,savefig,trace_data_path
from mcmc_tools import docontours_multi,save_traces
from agpy import pymc_plotting
import pymc_tools
import hopkins_pdf


# Hopkins - NO Mach number restrictions
d = mcmc_sampler_dict(tauoneone=tauoneone_hopkins,tautwotwo=tautwotwo_hopkins)
mc_hopkins_simple = pymc.MCMC(d)

graph_hopkins_simple = pymc.graph.graph(mc_hopkins_simple)
graph_hopkins_simple.write_pdf(savepath+"mc_hopkins_simple_graph.pdf")
graph_hopkins_simple.write_png(savepath+"mc_hopkins_simple_graph.png")

# Hopkins - with Mach number restrictions
d = mcmc_sampler_dict(tauoneone=tauoneone_hopkins,tautwotwo=tautwotwo_hopkins)
def Tval(sigma):
    return hopkins_pdf.T_of_sigma(sigma, logform=True)
d['Tval'] = pymc.Deterministic(name='Tval', eval=Tval, parents={'sigma':d['sigma']}, doc='Intermittency parameter T')
d['b'] = pymc.Uniform('b',0.3,1.0)
def mach(Tval, b):
    return 20*Tval / b
d['mach_mu'] = pymc.Deterministic(name='mach_mu', eval=mach, parents={'Tval':d['Tval'], 'b': d['b']}, doc='Mach Number')
d['mach'] = pymc.Normal(name='MachNumber',mu=d['mach_mu'], tau=0.5, value=5.1, observed=True)

mc_hopkins = pymc.MCMC(d)

# Hopkins - free Mach number
d = mcmc_sampler_dict(tauoneone=tauoneone_hopkins,tautwotwo=tautwotwo_hopkins)
def Tval(sigma):
    return hopkins_pdf.T_of_sigma(sigma, logform=True)
d['Tval'] = pymc.Deterministic(name='Tval', eval=Tval, parents={'sigma':d['sigma']}, doc='Intermittency parameter T')
d['b'] = pymc.Uniform('b',0.3,1.0)
def mach(Tval, b):
    return 20*Tval / b
d['mach'] = pymc.Deterministic(name='mach', eval=mach, parents={'Tval':d['Tval'], 'b': d['b']}, doc='Mach Number')
def mach_limits(m):
    if m < 1 or m > 50:
        return -np.inf
    else:
        return 0
d['sigma'].value = 2
d['mach_limits'] = pymc.Potential(name='mach_observed', logp=mach_limits, parents={'m':d['mach']},doc='Mach limits',verbose=0)

mc_hopkins_freemach = pymc.MCMC(d)

print "\nsimple hopkins sampling"
mc_hopkins_simple.sample(1e5)
print "\nhopkins sampling"
mc_hopkins.sample(1e5)
print "\nhopkins freemach sampling"
mc_hopkins_freemach.sample(1e5)


graph_hopkins = pymc.graph.graph(mc_hopkins)
graph_hopkins.write_pdf(savepath+"mc_hopkins_graph.pdf")
graph_hopkins.write_png(savepath+"mc_hopkins_graph.png")


def docontours_all(mc_hopkins_freemach,mc_hopkins,mc_hopkins_simple):
    docontours_multi(mc_hopkins_freemach,start=10000,savename=savepath+"mc_hopkins_freemach_multipanel.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','tau_ratio','mach','Tval','b'))
    docontours_multi(mc_hopkins_freemach,start=10000,savename=savepath+"mc_hopkins_freemach_multipanel_deviance.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','tau_ratio','mach','Tval','b','deviance'))

    docontours_multi(mc_hopkins,start=10000,savename=savepath+"mc_hopkins_withmach_multipanel.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','tau_ratio','Tval','b'))
    docontours_multi(mc_hopkins,start=10000,savename=savepath+"mc_hopkins_withmach_multipanel_deviance.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','tau_ratio','Tval','b','deviance'))
    docontours_multi(mc_hopkins_simple,start=10000,savename=savepath+"mc_hopkins_justtau_multipanel.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','tau_ratio','Tval'))
    docontours_multi(mc_hopkins_simple,start=10000,savename=savepath+"mc_hopkins_justtau_multipanel_deviance.pdf", dosave=True,
                     parnames=('tauoneone_mu','tautwotwo_mu','meandens','sigma','tau_ratio','Tval','deviance'))

docontours_all(mc_hopkins_freemach,mc_hopkins,mc_hopkins_simple)

if domillion:
    print "\nsimple hopkins sampling 1 million"
    mc_hopkins_simple.sample(1e6)
    print "\nhopkins sampling 1 million"
    mc_hopkins.sample(1e6)
    print "\nhopkins freemach sampling 1 million"
    mc_hopkins_freemach.sample(1e6)

    docontours_all(mc_hopkins_freemach,mc_hopkins,mc_hopkins_simple)

    #varslice=(1000,None,None)
#for fignum,(p1,p2) in enumerate(itertools.combinations(('tauoneone_mu','tautwotwo_mu','tau_ratio','sigma','meandens','Tval','b','mach_mu'),2)):
#    pymc_plotting.hist2d(mc_hopkins, p1, p2, bins=30, clear=True, fignum=fignum, varslice=varslice, colorbar=True)
#    pl.title("Hopkins with Mach")
#    pl.savefig(savepath+"HopkinsWithMach_%s_v_%s_mcmc.png" % (p1,p2))

#for fignum2,(p1,p2) in enumerate(itertools.combinations(('tauoneone_mu','tautwotwo_mu','tau_ratio','sigma','meandens'),2)):
#    pymc_plotting.hist2d(mc_hopkins_simple, p1, p2, bins=30, clear=True, fignum=fignum+fignum2+1, varslice=varslice, colorbar=True)
#    pl.title("Hopkins - just $\\tau$ fits")
#    pl.savefig(savepath+"HopkinsJustTau_%s_v_%s_mcmc.png" % (p1,p2))

print "Some statistics used in the paper: "
print 'mc_hopkins_simple sigma: ',mc_hopkins_simple.stats()['sigma']['quantiles']
print 'mc_hopkins        sigma: ',mc_hopkins.stats()['sigma']['quantiles']
print 'mc_hopkins        Tval: ',mc_hopkins.stats()['Tval']['quantiles']
print 'mc_hopkins        b: ',mc_hopkins.stats(quantiles=(0.1,1,2.5,5,50))['b']['quantiles']
print 'mc_hopkins        m: ',mc_hopkins.stats()['mach_mu']['quantiles']

hopkins_statstable = pymc_tools.stats_table(mc_hopkins)
hopkins_statstable.write(trace_data_path+'hopkins_statstable_abundance%s.fits' % abundance, overwrite=True)
hopkins_simple_statstable = pymc_tools.stats_table(mc_hopkins_simple)
hopkins_simple_statstable.write(trace_data_path+'hopkins_simple_statstable_abundance%s.fits' % abundance, overwrite=True)
hopkins_freemach_statstable = pymc_tools.stats_table(mc_hopkins_freemach)
hopkins_freemach_statstable.write(trace_data_path+'hopkins_freemach_statstable_abundance%s.fits' % abundance, overwrite=True)

save_traces(mc_hopkins, trace_data_path+"mc_hopkins_traces.fits", clobber=True)
save_traces(mc_hopkins_simple, trace_data_path+"mc_hopkins_simple_traces.fits", clobber=True)
save_traces(mc_hopkins_freemach, trace_data_path+"mc_hopkins_freemach_traces.fits", clobber=True)

pl.figure(32)
pl.clf()
pl.title("Hopkins")
pymc_plotting.plot_mc_hist(mc_hopkins,'b',lolim=True,alpha=0.5,bins=25,legloc='lower right')
savefig(savepath+'HopkinsWithMach_b_1D_restrictions.png')
