"""
Create a series of plots of sigma_s vs the measured things (tau, tau1/tau2)
"""
from measure_tau import trace_data_path, abundance, savefig, savepath
#import astropy.io.fits as pyfits
from agpy import pymc_plotting
import pylab as pl
import numpy as np

execfile('load_parameter_traces.py')

fig = pl.figure(1)
pl.clf()
ax1 = pymc_plotting.hist2d(mc_lognormal_freemach_traces, 'tauoneone_mu','sigma', bins=50,
                          clear=True, fignum=1, varslice=(2.5e5,None,None), colorbar=False,
                          axis=pl.subplot(231))
ax2 = pymc_plotting.hist2d(mc_lognormal_freemach_traces, 'tautwotwo_mu','sigma', bins=50,
                          clear=True, fignum=1, varslice=(2.5e5,None,None), colorbar=False,
                          axis=pl.subplot(232))
ax3 = pymc_plotting.hist2d(mc_lognormal_freemach_traces, 'tau_ratio','sigma', bins=50,
                          clear=True, fignum=1, varslice=(2.5e5,None,None), colorbar=False,
                          axis=pl.subplot(233))

ax4 = pymc_plotting.hist2d(mc_hopkins_freemach_traces, 'tauoneone_mu','sigma', bins=50,
                          clear=True, fignum=1, varslice=(2.5e5,None,None), colorbar=False,
                          axis=pl.subplot(234))
ax5 = pymc_plotting.hist2d(mc_hopkins_freemach_traces, 'tautwotwo_mu','sigma', bins=50,
                          clear=True, fignum=1, varslice=(2.5e5,None,None), colorbar=False,
                          axis=pl.subplot(235))
ax6 = pymc_plotting.hist2d(mc_hopkins_freemach_traces, 'tau_ratio','sigma', bins=50,
                          clear=True, fignum=1, varslice=(2.5e5,None,None), colorbar=False,
                          axis=pl.subplot(236))

pl.subplots_adjust(hspace=0,wspace=0)

for ii in (2,3,5,6):
    ax = pl.subplot(2,3,ii)
    ax.set_yticks([])
    ax.set_ylabel("")
for ii in (1,2,3):
    ax = pl.subplot(2,3,ii)
    ax.set_xticks([])
    ax.set_xlabel("")
for ii in (1,4):
    ax = pl.subplot(2,3,ii)
    ax.set_xlim(0.101,0.114)
    ax.set_xticks(np.round(np.linspace(*ax.get_xlim(),num=4)[1:],decimals=3))
    ax.set_ylabel("")
for ii in (2,5):
    ax = pl.subplot(2,3,ii)
    ax.set_xlim(0.0121,0.0187)
    ax.set_xticks(np.round(np.linspace(*ax.get_xlim(),num=4)[1:],decimals=3))
for ii in (3,6):
    ax = pl.subplot(2,3,ii)
    ax.set_xlim(5.4,8.64)

ax4.set_yticks(ax4.get_yticks()[:-2])
ax6.set_xticks(ax6.get_xticks()[2::2])

masterax = fig.add_axes( [0., 0., 1, 1] )
masterax.set_axis_off()
masterax.set_xlim(0, 1)
masterax.set_ylim(0, 1)
masterax.text( 
    .05, 0.5, r'$\sigma_s$', rotation='vertical',
    horizontalalignment='center', verticalalignment='center',
    fontsize=24)

ax4.set_xlabel(r'$\tau_{1-1}$',fontsize=24)
ax5.set_xlabel(r'$\tau_{2-2}$',fontsize=24)
ax6.set_xlabel(r'$\tau_{1-1}/\tau_{2-2}$',fontsize=24)

# ax.hlines([3.68],*ax.get_xlim(),linestyles=['--'], colors=['k'], label="H$_2$CO Lower-limit")
# ax.hlines([6.58],*ax.get_xlim(),linestyles=[':'],  colors=['k'], label="CO Lower-limit")
# 
# ax.legend(loc='best')
# 
# ax.set_ylabel(r"$\mathcal{M}_{3D}$")
# ax.set_xlabel('$\\tau_{1-1}/\\tau_{2-2}$',fontsize=24)

savefig(savepath+"sigma_vs_tauratio_sixpanels.png")

