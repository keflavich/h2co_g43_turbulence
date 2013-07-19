import astropy.io.fits as pyfits
from measure_tau import abundance,trace_data_path

if not 'lognormal_statstable' in locals():
    lognormal_statstable = pyfits.getdata(trace_data_path+'lognormal_statstable_abundance%s.fits' % abundance)
    lognormal_simple_statstable = pyfits.getdata(trace_data_path+'lognormal_simple_statstable_abundance%s.fits' % abundance)
    lognormal_freemach_statstable = pyfits.getdata(trace_data_path+'lognormal_freemach_statstable_abundance%s.fits' % abundance)
if not 'hopkins_statstable' in locals():
    hopkins_statstable = pyfits.getdata(trace_data_path+'hopkins_statstable_abundance%s.fits' % abundance)
    hopkins_simple_statstable = pyfits.getdata(trace_data_path+'hopkins_simple_statstable_abundance%s.fits' % abundance)
    hopkins_freemach_statstable = pyfits.getdata(trace_data_path+'hopkins_freemach_statstable_abundance%s.fits' % abundance)

# clearly not done yet
one_sided = ['b']
two_sided = ['sigma','Tval']
tex = {'b':r'$b$', 'sigma': r'$\sigma_s | M$', 'Tval':r'$T$', 'sigmab':r'$\sigma_s$'}
fmt = {'b':r'$>%0.2f$', 'sigma': r'%0.1f', 'Tval':r'%0.2f', 'sigmab':r'%0.1f'}
with open('distribution_fit_table.tex','w') as f:
    for v in one_sided:
        line = [tex[v]]
        for table in [lognormal_statstable,hopkins_statstable,]:
            if v in table['variable name']:
                row = table[table['variable name']==v]
                #line += ["%0.1g" % x for x in (row['q0.1'],row['q1.0'],row['q5.0'])]
                #line += ["-",fmt[v] % row['q5.0']]
                line += [fmt[v] % row['q5.0']]
            else:
                line += ["-"]# * 2
        print >>f,"&".join(line),r"\\"
    v='sigma'
    line = [tex['sigmab']]
    for table in [lognormal_simple_statstable,hopkins_simple_statstable]:
        if v in table['variable name']:
            row = table[table['variable name']==v]
            #line += [fmt[v] % row['q50.0'], ("$^{"+fmt[v]+"}_{"+fmt[v]+"}$") % (row['q2.5'],row['q97.5'])]
            line += [(fmt[v] % row['q50.0']) + ("$^{"+fmt[v]+"}_{"+fmt[v]+"}$") % (row['q50.0']-row['q2.5'],row['q97.5']-row['q50.0'])]
        else:
            line += ["-"]# * 2
    print >>f,"&".join(line),r"\\"
    for v in two_sided:
        line = [tex[v]]
        for table in [lognormal_statstable,hopkins_statstable,]:
            if v in table['variable name']:
                row = table[table['variable name']==v]
                #line += [fmt[v] % row['q50.0'], ("$^{"+fmt[v]+"}_{"+fmt[v]+"}$") % (row['q2.5'],row['q97.5'])]
                line += [(fmt[v] % row['q50.0']) + ("$^{"+fmt[v]+"}_{"+fmt[v]+"}$") % (row['q50.0']-row['q2.5'],row['q97.5']-row['q50.0'])]
            else:
                line += ["-"]# * 2
        print >>f,"&".join(line),r"\\"
