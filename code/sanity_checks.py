import pylab as pl
import numpy as np
from measure_tau import tau11,etau11,tau22,etau22,ratio,eratio,tauoneone_hopkins,tautwotwo_hopkins,tauoneone,tautwotwo
import progressbar
pb = progressbar.ProgressBar()

npix = 50
pars11 = np.empty([npix,npix])
pars22 = np.empty([npix,npix])
ratiomap = np.empty([npix,npix])
logpars11 = np.empty([npix,npix])
logpars22 = np.empty([npix,npix])
logratiomap = np.empty([npix,npix])

densmin,densmax = 5,300
sigmin,sigmax = 0.05,3.0

densities = np.linspace(densmin,densmax,npix)
sigmas = np.linspace(sigmin,sigmax,npix)
pb.start()
for i,dens in enumerate(densities):
    for j,sigma in enumerate(sigmas):
        pars11[j,i] = tauoneone_hopkins(dens,sigma)
        pars22[j,i] = tautwotwo_hopkins(dens,sigma)
        ratiomap[j,i] = pars11[j,i]/pars22[j,i]
        logpars11[j,i] = tauoneone(dens,sigma)
        logpars22[j,i] = tautwotwo(dens,sigma)
        logratiomap[j,i] = logpars11[j,i]/logpars22[j,i]
    pb.update(i)
pb.finish()

escale = 2 # bigger error

pl.figure(1)
pl.clf()
pl.suptitle('Hopkins')
pl.subplot(221)
pl.pcolormesh(densities,sigmas,pars11,vmin=0,vmax=0.3)
pl.title('1-1')
pl.colorbar()
pl.contour(densities,sigmas,pars11,levels=[tau11-etau11*escale,tau11+etau11*escale],colors=['w']*2)
pl.xlabel('Density')
pl.ylabel(r'$\sigma$')
pl.subplot(222)
pl.pcolormesh(densities,sigmas,pars22,vmin=0,vmax=0.3)
pl.colorbar()
pl.contour(densities,sigmas,pars22,levels=[tau22-etau22*escale,tau22+etau22*escale],colors=['k']*2)
pl.xlabel('Density')
pl.ylabel(r'$\sigma$')
pl.title('2-2')
pl.subplot(223)
pl.pcolormesh(densities,sigmas,ratiomap)
pl.colorbar()
pl.contour(densities,sigmas,ratiomap,levels=[ratio-eratio*escale,ratio+eratio*escale],colors=['m']*2)
pl.xlabel('Density')
pl.ylabel(r'$\sigma$')
pl.title('1-1/2-2')
pl.subplot(224)
pl.contour(densities,sigmas,pars11,levels=[tau11-etau11*escale,tau11+etau11*escale],colors=['b']*2)
pl.contour(densities,sigmas,pars22,levels=[tau22-etau22*escale,tau22+etau22*escale],colors=['k']*2)
pl.contour(densities,sigmas,ratiomap,levels=[ratio-eratio*escale,ratio+eratio*escale],colors=['m']*2)
pl.xlabel('Density')
pl.ylabel(r'$\sigma$')
pl.title('Just fitted contours')

pl.figure(2)
pl.clf()
pl.suptitle('Lognormal')
pl.subplot(221)
pl.pcolormesh(densities,sigmas,logpars11,vmin=0,vmax=0.3)
pl.colorbar()
pl.contour(densities,sigmas,logpars11,levels=[tau11-etau11*escale,tau11+etau11*escale],colors=['w']*2)
pl.xlabel('Density')
pl.ylabel(r'$\sigma$')
pl.title('1-1')
pl.subplot(222)
pl.pcolormesh(densities,sigmas,logpars22,vmin=0,vmax=0.3)
pl.colorbar()
pl.contour(densities,sigmas,logpars22,levels=[tau22-etau22*escale,tau22+etau22*escale],colors=['k']*2)
pl.xlabel('Density')
pl.ylabel(r'$\sigma$')
pl.title('2-2')
pl.subplot(223)
pl.pcolormesh(densities,sigmas,logratiomap)
pl.colorbar()
pl.contour(densities,sigmas,logratiomap,levels=[ratio-eratio*escale,ratio+eratio*escale],colors=['m']*2)
pl.xlabel('Density')
pl.ylabel(r'$\sigma$')
pl.title('1-1/2-2')
pl.subplot(224)
pl.contour(densities,sigmas,logpars11,levels=[tau11-etau11*escale,tau11+etau11*escale],colors=['b']*2)
pl.contour(densities,sigmas,logpars22,levels=[tau22-etau22*escale,tau22+etau22*escale],colors=['k']*2)
pl.contour(densities,sigmas,logratiomap,levels=[ratio-eratio*escale,ratio+eratio*escale],colors=['m']*2)
pl.xlabel('Density')
pl.ylabel(r'$\sigma$')
pl.title('Just fitted contours')
