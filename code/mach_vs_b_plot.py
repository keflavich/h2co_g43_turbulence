from measure_tau import trace_data_path, abundance, savefig, savepath
import astropy.io.fits as pyfits
from agpy import pymc_plotting
#import pylab as pl

for abundance in (-8.5,-9.0):

    lognormal_freemach_statstable = pyfits.getdata(trace_data_path+'lognormal_freemach_statstable_abundance%s.fits' % abundance)
    mc_lognormal_freemach_traces = pyfits.getdata(trace_data_path+"mc_lognormal_freemach_traces%s.fits" % abundance)


    ax = pymc_plotting.hist2d(mc_lognormal_freemach_traces, 'b','mach', bins=50,
                              clear=True, fignum=1, varslice=(2.5e5,None,None), colorbar=True)

    ax.hlines([3.68],*ax.get_xlim(),linestyles=['--'], colors=['k'], label="H$_2$CO")
    ax.hlines([6.58],*ax.get_xlim(),linestyles=[':'],  colors=['k'], label="CO")

    ax.legend(loc='best')

    ax.set_ylabel(r"$\mathcal{M}_{3D}$")
    ax.set_xlabel('$b$',fontsize=24)

    savefig(savepath+"mach_vs_b_lognormal_mcmc_contours_abundance%s.png" % abundance)


    mc_hopkins_freemach_traces = pyfits.getdata(trace_data_path+"mc_hopkins_freemach_traces%s.fits" % abundance)


    ax = pymc_plotting.hist2d(mc_hopkins_freemach_traces, 'b','mach', bins=50,
                              clear=True, fignum=1, varslice=(2.5e5,None,None), colorbar=True)

    ax.hlines([3.68],*ax.get_xlim(),linestyles=['--'], colors=['k'], label="H$_2$CO")
    ax.hlines([6.58],*ax.get_xlim(),linestyles=[':'],  colors=['k'], label="CO")

    ax.legend(loc='best')

    ax.set_ylabel(r"$\mathcal{M}_{3D}$")
    ax.set_xlabel('$b$',fontsize=24)

    savefig(savepath+"mach_vs_b_hopkins_mcmc_contours_abundance%s.png" % abundance)

