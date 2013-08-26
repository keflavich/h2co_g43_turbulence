import pylab as pl
import numpy as np
from smoothtau_models import generate_simpletools,select_data
from measure_tau import savefig,savepath,tau11,etau11,tau22,etau22,ratio,eratio
import itertools

for abundance in [-9,-8.5,-8.25,-8.0]:
    opr = 1
    temperature = 20
    tau1x,tau2x,dens,col = select_data(abundance=abundance, opr=opr, temperature=temperature)
    tauratio,tauratio_hopkins,tau,tau_hopkins = generate_simpletools(abundance=abundance, opr=opr, temperature=temperature)
    print len(tau1x)

    pl.figure(30)
    pl.clf()
    ax = pl.gca()
    ax.plot(dens,tau1x/tau2x,'k',linewidth=3.0,label=r'Dirac $\delta$',alpha=0.75)

    logmeandens = np.linspace(-2,7,300)
    meandens = 10**logmeandens

    stylecycle = itertools.cycle(('-','-.','--',':'))
    dashcycle = itertools.cycle(((None,None),(6,2),(10,4),(2,2),(5,5)))

    for sigma in np.arange(0.5,4.0,1):
        ax.plot(logmeandens,tauratio(meandens,sigma=sigma),color='k',linewidth=2, alpha=0.5,  label='$\\sigma_s=%0.1f$' % sigma, dashes=dashcycle.next())

    dashcycle = itertools.cycle(((None,None),(6,2),(10,4),(2,2),(5,5)))
    for sigma in np.arange(0.5,4.0,1):
        ax.plot(logmeandens,tauratio_hopkins(meandens,sigma=sigma),color='orange', label='$\\sigma_s=%0.1f$ Hopkins' % sigma, linewidth=2, alpha=0.5, dashes=dashcycle.next())

    ax.legend(loc='best',prop={'size':18})
    ax.axis([-1,7,0,15])
    ax.set_xlabel('$\\log_{10}$($\\langle\\rho\\rangle_V($H$_2)$ [cm$^{-3}$])',fontsize=24)
    ax.set_ylabel('$\\tau_{1-1}/\\tau_{2-2}$',fontsize=24)
    savefig(savepath+'lognormalsmooth_density_ratio_massweight_withhopkins_logopr%0.1f_abund%s.png' % (np.log10(opr),str(abundance)),bbox_inches='tight')

    dot,caps,bars = ax.errorbar([np.log10(30)],
                                [ratio],
                                xerr=np.array([[0.47,0.82]]).T,
                                yerr=[eratio], # np.array([[0.87,1.11]]).T,
                                label="G43.17+0.01",
                                color=(0,0,1,0.5),
                                alpha=0.5,
                                marker='o',
                                linewidth=2)
    caps[0].set_marker('$($')
    caps[1].set_marker('$)$')
    caps[0].set_color((1,0,0,0.6))
    caps[1].set_color((1,0,0,0.6))
    bars[0].set_color((1,0,0,0.6))
    savefig(savepath+'lognormalsmooth_density_ratio_massweight_withhopkins_logopr%0.1f_abund%s_withG43.png' % (np.log10(opr),str(abundance)),bbox_inches='tight')

    for ii,(line,taux) in enumerate(zip(('oneone','twotwo'),(tau1x,tau2x))):
        pl.figure(31+ii)
        pl.clf()
        ax = pl.gca()
        ax.semilogy(dens,taux,'k',linewidth=3.0,label=r'Dirac $\delta$',alpha=0.75)

        logmeandens = np.linspace(-2,7,300)
        meandens = 10**logmeandens

        stylecycle = itertools.cycle(('-','-.','--',':'))
        dashcycle = itertools.cycle(((None,None),(6,2),(10,4),(2,2),(5,5)))

        for sigma in np.arange(0.5,4.0,1):
            ax.semilogy(logmeandens,tau(meandens,sigma=sigma,line=taux),color='k',linewidth=2, alpha=0.5,  label='$\\sigma_s=%0.1f$' % sigma, dashes=dashcycle.next())

        dashcycle = itertools.cycle(((None,None),(6,2),(10,4),(2,2),(5,5)))
        for sigma in np.arange(0.5,4.0,1):
            ax.semilogy(logmeandens,tau_hopkins(meandens,sigma=sigma,line=taux),color='orange', label='$\\sigma_s=%0.1f$ Hopkins' % sigma, linewidth=2, alpha=0.5, dashes=dashcycle.next())

        ax.legend(loc='best',prop={'size':18})
        ax.axis([-2,7,1e-3,10])
        ax.set_xlabel('$\\log_{10}$($\\langle\\rho\\rangle_V($H$_2)$ [cm$^{-3}$])',fontsize=24)
        linelabel = {'oneone':'1-1','twotwo':'2-2'}[line]
        ax.set_ylabel('$\\tau_{%s}$' % linelabel,fontsize=24)
        savefig(savepath+'lognormalsmooth_density_tau_%s_massweight_withhopkins_logopr%0.1f_abund%s.png' % (line, np.log10(opr),str(abundance)),bbox_inches='tight')

        tau_meas = {'oneone': [tau11,etau11], 'twotwo':[tau22,etau22]}
        dot,caps,bars = ax.errorbar([np.log10(30)],
                                    tau_meas[line][0],
                                    xerr=np.array([[0.47,0.82]]).T,
                                    yerr=tau_meas[line][1]*3,
                                    label="G43.17+0.01",
                                    color=(0,0,1,0.5),
                                    alpha=0.5,
                                    marker='o',
                                    linewidth=2)
        caps[0].set_marker('$($')
        caps[1].set_marker('$)$')
        caps[0].set_color((1,0,0,0.6))
        caps[1].set_color((1,0,0,0.6))
        bars[0].set_color((1,0,0,0.6))
        savefig(savepath+'lognormalsmooth_density_tau_%s_massweight_withhopkins_logopr%0.1f_abund%s_withG43.png' % (line, np.log10(opr),str(abundance)),bbox_inches='tight')
