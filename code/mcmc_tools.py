import agpy
import multiplot
import pylab as pl
import numpy as np
import astropy.io.fits as pyfits
import itertools

def savefig(savename, **kwargs):
    pl.savefig(savename.replace("pdf","png"), **kwargs)
    pl.savefig(savename.replace("png","pdf"), **kwargs)

# pymc tool
def save_traces(mc, filename, clobber=False):
    keys = [v.__name__ for v in mc.variables if v.__name__ in mc.db._traces]
    traces = {k:np.concatenate([v.squeeze()
                                for v in mc.trace(k)._trace.values()])
              for k in keys}
    shape = len(traces[keys[0]])
    arr = np.empty(shape, dtype=[(k,np.float) for k in keys])
    for k in keys:
        arr[k] = traces[k]
    hdu = pyfits.BinTableHDU(arr)
    hdu.writeto(filename, clobber=clobber)
    return traces

def docontours_multi(mc, start=6000, end=None, skip=None, dosave=False,
                     parnames=(), bins=30, fignum=35,
                     savename='multipanel.png'):
    if isinstance(mc, dict):
        traces = mc
    else:
        mcnames = mc.db._traces.keys()
        traces = {k:np.concatenate([v.squeeze()
                  for v in mc.trace(k)._trace.values()])[start:end:skip]
                  for k in mcnames
                  if k in parnames}

    pl.figure(fignum)
    pl.clf()
    npars = len(traces.keys())
    mp = multiplot.multipanel(dims=(npars,npars), padding=(0,0), diagonal=True, figID=fignum)
    parnumbers = {p:i for i,p in enumerate(traces.keys())}
    for par1,par2 in itertools.combinations(traces.keys(),2):
        # coordinates are -y, x
        axno = mp.axis_number(parnumbers[par2],parnumbers[par1])
        if axno is None:
            continue
        ax = mp.grid[axno]
        try:
            agpy.pymc_plotting.hist2d(traces, par1, par2,
                                      varslice=(None,None,None), axis=ax,
                                      colorbar=False, clear=True,
                                      doerrellipse=False, bins=bins)
        except ValueError as E:
            print E
            continue
        # if x > 0, hide y labels.
        if parnumbers[par1] > 0:
            ax.set_ylabel("")
            ax.set_yticks([])
        else:
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.yaxis.get_major_locator().set_params(nbins=4)
        if parnumbers[par2] == npars-1:
            ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.xaxis.get_major_locator().set_params(nbins=4)
            ax.tick_params(axis='both', which='major', labelsize=12)
        else:
            ax.set_xlabel('')
            ax.set_xticks([])
    for par in traces.keys():
        ax = mp.grid[mp.axis_number(parnumbers[par],parnumbers[par])]
        try:
            agpy.pymc_plotting.plot_mc_hist(traces, par,
                                            varslice=(None,None,None),
                                            onesided=False, axis=ax,
                                            legend=False)
        except Exception as E:
            print E
            continue
        # if x > 0, hide y labels.
        if parnumbers[par] > 0:
            ax.set_ylabel("")
            ax.set_yticks([])
        else:
            ax.set_ylabel(par, fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.yaxis.get_major_locator().set_params(nbins=4)
        if npars-1 == parnumbers[par]: # label bottom panel
            ax.set_xlabel(par, fontsize=12)
            ax.xaxis.get_major_locator().set_params(nbins=4)
            ax.tick_params(axis='both', which='major', labelsize=12)

    if dosave:
        savefig(savename,bbox_inches='tight')
