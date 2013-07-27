import pylab as pl
import numpy as np
import hopkins_pdf
from turbulent_pdfs import lognormal

def normalize(arr, normfunc=np.sum):
    return arr/normfunc(arr)

pl.rc('font',size=24)
meandens_4 = 15
xdens = np.linspace(-6,10,1000)
for fignum,sigma in enumerate((0.5,1,1.5,2.0,2.5)):

    rho = 10**xdens
    distr = lognormal(dens=rho, meandens=meandens_4, sigma=sigma)

    pl.figure(fignum)
    pl.clf()
    #pl.plot(rho,distr_mass,label='Mass-weighted PDF',linewidth=2.0)
    pl.plot(rho,normalize(distr),    label='Lognormal $P_V$',color='k', linestyle='-')
    pl.plot(rho,normalize(rho*distr),label='Lognormal $P_M$',color='k', linestyle='--')
    #pl.plot(10**(xdens+log(10)),distr_mass,":",label='Volume-weighted PDF scaled')
    #pl.plot(rho,massgauss,'--',label='Mass gaussian')
    #pl.plot(rho,taildens,'k--',label='with High Tail')
    #pl.plot(rho,taildenspower,'k:',label='with powerlaw Tail')
    #pl.plot(rho,powertail,label='powerlaw Tail')
    #pl.plot(rho,tail,label='High Tail')
    #pl.loglog(rho,hightail_distr(meandens_4,sigma,dens=xdens),label='High Tail',linewidth=2, alpha=0.5)
    #pl.loglog(rho,lowtail_distr(meandens_4,sigma,dens=xdens),label='Low Tail',linewidth=2, alpha=0.5)
    #pl.loglog(rho,compressive_distr(meandens_4,sigma,dens=xdens),label='Compressive',linewidth=2, alpha=0.5)
    #pl.loglog(rho,compressive_distr(meandens_4,sigma,dens=xdens,sigma2=sigma*0.6,secondscale=1.2,offset=1.9),label='Compressive2',linewidth=3, alpha=0.5)
    pl.loglog(rho, normalize(hopkins_pdf.hopkins(rho, meanrho=(meandens_4), sigma=sigma, T=hopkins_pdf.T_of_sigma(sigma))),
              label='Hopkins $P_V$',linewidth=3, alpha=0.5, linestyle='-', color='orange')
    pl.loglog(rho,normalize(rho*hopkins_pdf.hopkins(rho, meanrho=(meandens_4), sigma=sigma, T=hopkins_pdf.T_of_sigma(sigma))),
              label='Hopkins $P_M$',linewidth=3, alpha=0.5, linestyle='--', color='orange')
    #pl.loglog(rho,hopkins_pdf.hopkins(10**(xdens), meanrho=10**(meandens_4), sigma=sigma, T=0.1),label='Hopkins T=0.1',linewidth=3, alpha=0.5)
    #hopkins_pdf.hopkins_masspdf_ofmeandens(np.exp(dens), np.exp(meandens), T) # T~0.05 M_C
    pl.legend(loc='upper left',prop={'size':19.5})
    pl.vlines([15,1e4],*pl.gca().get_ylim(), color='b', linestyle=':')
    pl.xlabel(r'$\rho(H_2)$ cm$^{-3}$',fontsize=30)
    pl.ylabel(r'$P(\rho)$',fontsize=30)
    pl.title('$\\sigma=%0.1f$' % sigma)
    pl.gca().set_xlim(1e-4,1e7)
    pl.gca().set_ylim(1e-5,pl.gca().get_ylim()[1])
    pl.savefig('/Users/adam/work/h2co/lowdens/figures/lognormalsmooth_density_distributions_sigma%0.1f.png' % sigma,bbox_inches='tight')
    pl.savefig('/Users/adam/work/h2co/lowdens/figures/lognormalsmooth_density_distributions_sigma%0.1f.pdf' % sigma,bbox_inches='tight')

pl.show()
