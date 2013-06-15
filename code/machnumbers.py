import numpy as np

for linewidth in [0.95, 1.7]:
    # linewidth = 1.7 # km/s fwhm
    cs = 0.19
    M1D = linewidth / (8*np.log(2))**0.5 / cs
    M3D = 3**0.5 * M1D
    print "linewidth: %f, M1D: %f, M3D: %f" % (linewidth, M1D, M3D)
