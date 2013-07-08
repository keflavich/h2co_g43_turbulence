from measure_tau import trace_data_path, abundance, savefig, savepath
import astropy.io.fits as pyfits
from agpy import pymc_plotting
#import pylab as pl

lognormal_freemach_statstable = pyfits.getdata(trace_data_path+'lognormal_freemach_statstable_abundance%s.fits' % abundance)
mc_lognormal_freemach_traces = pyfits.getdata(trace_data_path+"mc_lognormal_freemach_traces.fits")


ax = pymc_plotting.hist2d(mc_lognormal_freemach_traces, 'tau_ratio','mach', bins=50,
                          clear=True, fignum=1, varslice=(None,None,None), colorbar=True)

ax.hlines([3.68],*ax.get_xlim(),linestyles=['--'], colors=['k'], label="H$_2$CO Lower-limit")
ax.hlines([6.58],*ax.get_xlim(),linestyles=[':'],  colors=['k'], label="CO Lower-limit")

ax.legend(loc='best')

ax.set_ylabel(r"$\mathcal{M}_{3D}$")
ax.set_xlabel('$\\tau_{1-1}/\\tau_{2-2}$',fontsize=24)

savefig(savepath+"mach_vs_tauratio_lognormal_mcmc_contours.png")
