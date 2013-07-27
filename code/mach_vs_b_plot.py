from measure_tau import trace_data_path, savefig, savepath
import astropy.io.fits as pyfits
from agpy import pymc_plotting
import pylab as pl

pl.rc('font',size=30)

for abundance in (-8.5,-9.0):
    for withfree in ('with','free'):

        logtraces = pyfits.getdata(trace_data_path+"mc_lognormal_%smach_traces%s.fits" % (withfree,abundance))

        ax = pymc_plotting.hist2d(logtraces, 'b','mach', bins=50,
                                  clear=True, fignum=1, varslice=(2.5e5,None,None), colorbar=True)

        ax.hlines([3.68],*ax.get_xlim(),linestyles=['--'], colors=['k'], label="H$_2$CO")
        ax.hlines([6.58],*ax.get_xlim(),linestyles=[':'],  colors=['k'], label="CO")

        ax.legend(loc='best')

        ax.set_ylabel(r"$\mathcal{M}_{3D}$")
        ax.set_xlabel('$b$',fontsize=36)

        savefig(savepath+"mach_vs_b_lognormal_mcmc_contours_%smach_abundance%s.png" % (withfree,abundance))


        hoptraces = pyfits.getdata(trace_data_path+"mc_hopkins_%smach_traces_abundance%s.fits" % (withfree,abundance))

        if 'mach' in hoptraces.names:
            mach = 'mach'
        else:
            mach = 'mach_mu'

        ax = pymc_plotting.hist2d(hoptraces, 'b',mach, bins=50,
                                  clear=True, fignum=1, varslice=(2.5e5,None,None), colorbar=True)

        ax.hlines([3.68],*ax.get_xlim(),linestyles=['--'], colors=['k'], label="H$_2$CO")
        ax.hlines([6.58],*ax.get_xlim(),linestyles=[':'],  colors=['k'], label="CO")

        ax.legend(loc='best')

        ax.set_ylabel(r"$\mathcal{M}_{3D}$")
        ax.set_xlabel('$b$',fontsize=36)

        savefig(savepath+"mach_vs_b_hopkins_mcmc_contours_%smach_abundance%s.png" % (withfree,abundance))

