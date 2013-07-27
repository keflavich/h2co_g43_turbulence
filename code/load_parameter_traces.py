from measure_tau import trace_data_path
import astropy.io.fits as pyfits

if 'abundance' not in locals():
    from measure_tau import abundance

hopkins_statstable = pyfits.getdata(trace_data_path+'hopkins_statstable_abundance%s.fits' % abundance)
hopkins_simple_statstable = pyfits.getdata(trace_data_path+'hopkins_simple_statstable_abundance%s.fits' % abundance)
hopkins_freemach_statstable = pyfits.getdata(trace_data_path+'hopkins_freemach_statstable_abundance%s.fits' % abundance)
lognormal_statstable = pyfits.getdata(trace_data_path+'lognormal_statstable_abundance%s.fits' % abundance)
lognormal_simple_statstable = pyfits.getdata(trace_data_path+'lognormal_simple_statstable_abundance%s.fits' % abundance)
lognormal_freemach_statstable = pyfits.getdata(trace_data_path+'lognormal_freemach_statstable_abundance%s.fits' % abundance)

mc_hopkins_traces = pyfits.getdata(trace_data_path+"mc_hopkins_traces_abundance%s.fits" % abundance)
mc_hopkins_simple_traces = pyfits.getdata(trace_data_path+"mc_hopkins_simple_traces%s.fits" % abundance)
mc_hopkins_freemach_traces = pyfits.getdata(trace_data_path+"mc_hopkins_freemach_traces%s.fits" % abundance)
mc_lognormal_traces = pyfits.getdata(trace_data_path+"mc_lognormal_withmach_traces%s.fits" % abundance)
mc_lognormal_simple_traces = pyfits.getdata(trace_data_path+"mc_lognormal_simple_traces%s.fits" % abundance)
mc_lognormal_freemach_traces = pyfits.getdata(trace_data_path+"mc_lognormal_freemach_traces%s.fits" % abundance)
