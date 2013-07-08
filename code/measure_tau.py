from smoothtau_models import generate_simpletools,select_data
import pylab as pl
import pymc

trace_data_path = '/Users/adam/work/h2co/lowdens/code/trace_data/'

savepath = "/Users/adam/work/h2co/lowdens/figures/"
# load defaults by default
abundance = -8.5

tauratio,tauratio_hopkins,tau,tau_hopkins = generate_simpletools(abundance=abundance)
tau1x,tau2x,dens,col = select_data(abundance=abundance)
def tauoneone(meandens, sigma):
    return tau(meandens, sigma, line=tau1x)
def tautwotwo(meandens, sigma):
    return tau(meandens, sigma, line=tau2x)
def tauoneone_hopkins(meandens, sigma):
    return tau_hopkins(meandens, sigma, line=tau1x)
def tautwotwo_hopkins(meandens, sigma):
    return tau_hopkins(meandens, sigma, line=tau2x)

def mcmc_sampler_dict(tauoneone=tauoneone,tautwotwo=tautwotwo,truncate_at_50sigma=False):
    """
    Generator for the MCMC parameters

    truncate_at_50sigma will reject all "solutions" that are 5-sigma deviant
    from the measured optical depths
    """
    d = {}
    # fit values for GSRMC 43.30
    # meandens is "observed" but we want to trace it and let it vary...
    # with meandens observed=False,  'sigma': {'95% HPD interval': array([ 2.89972948,  3.69028675]),
    d['meandens'] = pymc.Uniform(name='meandens',lower=10,upper=200,value=60, observed=False)
    d['sigma'] = pymc.Uniform(name='sigma',lower=0,upper=25,value=2.88)
    # the observed values.  f=tau ratio = 6.65 (6.99?).  tau might be too high, but the "best fits" were tau=9 before, which is just not possible
    tau11 = 0.1133
    etau11=0.001165
    tau22 = 0.01623
    etau22 = 0.000525
    d['tauoneone_mu'] = pymc.Deterministic(name='tauoneone_mu', eval=tauoneone, parents={'meandens':d['meandens'],'sigma':d['sigma']}, doc='tauoneone')
    d['tautwotwo_mu'] = pymc.Deterministic(name='tautwotwo_mu', eval=tautwotwo, parents={'meandens':d['meandens'],'sigma':d['sigma']}, doc='tautwotwo')
    if truncate_at_50sigma:
        d['sigma'].value = 1.86
        d['meandens'].value = 60
        d['tauoneone'] = pymc.TruncatedNormal(name='tauoneone',mu=d['tauoneone_mu'],tau=1./etau11**2,value=tau11,
                                              a=tau11-50*etau11,b=tau11+50*etau11, observed=True)
        d['tautwotwo'] = pymc.TruncatedNormal(name='tautwotwo',mu=d['tautwotwo_mu'],tau=1./etau22**2,value=tau22,
                                              a=tau22-50*etau22,b=tau22+50*etau22, observed=True)
    else:
        d['tauoneone'] = pymc.Normal(name='tauoneone',mu=d['tauoneone_mu'],tau=1./etau11**2,value=tau11,observed=True)
        d['tautwotwo'] = pymc.Normal(name='tautwotwo',mu=d['tautwotwo_mu'],tau=1./etau22**2,value=tau22,observed=True)
    @pymc.deterministic(trace=True,plot=True)
    def tau_ratio(oneone=d['tauoneone_mu'], twotwo=d['tautwotwo_mu']):
        return oneone/twotwo
    d['tau_ratio'] = tau_ratio
    return d

def savefig(savename, **kwargs):
    pl.savefig(savename.replace("pdf","png"), **kwargs)
    pl.savefig(savename.replace("png","pdf"), **kwargs)

T,F = True,False
domillion=T

if __name__ == "__main__":
    execfile('lognormal_mcmc_parameter_estimates.py')
    execfile('hopkins_mcmc_parameter_estimates.py')
    execfile('tau_figures.py')
    execfile('mcmc_parameter_estimate_tables.py')
