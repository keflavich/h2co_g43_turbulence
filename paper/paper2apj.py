import re,os,time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--reconvert",default=False,action='store_true')
parser.add_argument("--arxiv",default=False,action='store_true')
args = parser.parse_args()
print "ARGS: ",args

ppath='/Users/adam/work/h2co/lowdens/paper/'
paper_name = 'h2co_turbulence_letter'
file = open(ppath+paper_name+'.tex','r')

outdir = "apj/" if not args.arxiv else "arxiv/"
outtype = "apj" if not args.arxiv else "arxiv"

if not os.path.exists(ppath+outdir):
    os.mkdir(ppath+outdir)

outfn = ppath+outtype+'form_temp.tex'
outf = open(outfn,'w')

inputre = re.compile('input{(.*)}')
includere = re.compile('include{(.*)}')
bibre   = re.compile('bibliography{(.*)}')
emulateapjre = re.compile("documentclass{emulateapj}")
#                           \documentclass{emulateapj}
beginre = re.compile('\\begin{document}')
endre = re.compile('\\end{document}')
prefacere = re.compile('\\input{preface.*}')
solobibre = re.compile("\\input{solobib}")

def strip_input(list_of_lines):
    # strip out preface, solobib, end{doc}
    #return "".join(list_of_lines[1:-2])
    return "".join(
            [line for line in list_of_lines 
            if not prefacere.search(line)
            and not solobibre.search(line)
            and not endre.search(line)
            and not beginre.search(line)]
            )


for line in file.readlines():
    if line[0] == "%":
        continue
    input = inputre.search(line)
    include = includere.search(line)
    bib = bibre.search(line)
    if input is not None:
        fn = input.groups()[0] + ".tex"
        print "Doing input " + fn
        f = open(ppath+fn,'r')
        if 'preface' in line:
            print >>outf,f.read(),
        else:
            print >>outf,strip_input(f.readlines()),
        f.close()
    elif include is not None:
        fn = include.groups()[0] + ".tex"
        print "Doing include " + fn
        f = open(ppath+fn,'r')
        if 'preface' in line:
            print >>outf,f.read(),
        else:
            print >>outf,strip_input(f.readlines()),
        f.close()
    elif bib is not None:
        bn = bib.groups()[0] + '.bbl'
        print "Doing bibliography " + bn
        f = open(ppath+bn,'r')
        print >>outf,strip_input(f.readlines()),
        f.close()
    else:
        print >>outf,line,

outf.close()
file.close()

file = open(outfn,'r')
outfn = ppath+outtype+'form.tex'
outf = open(outfn,'w')

default_suffix='.png'
figre = re.compile('Figure{{?(.*?)}?}')
fig2re = re.compile('FigureTwoA?A?{{?(.*?)}?}\n?\s*{(.*?)}?}',flags=re.MULTILINE)
fig2re1line = re.compile('FigureTwoA?A?{{?(.*?)}?}',flags=re.MULTILINE)
fig4re = re.compile('FigureFourP?D?F?\n?\s*{{?(.*?)}?}\n\s*?{{?(.*?)}}?\n\s*?{{?(.*?)}}?\n\s*?{{?(.*?)}}?',flags=re.MULTILINE)
fig4re1line = re.compile('FigureFourP?D?F?({{?(.*?)}?})?',flags=re.MULTILINE)
plotonere = re.compile('plotone{{?(.*?)}')
includegre = re.compile('includegraphics\[.*\]{{?(.*?)}?}')
fig_suffixes = "png|eps|pdf"
lonelygraphics = re.compile('^\s*{{?(.*?)}?}')

out_suffix = "eps" if not args.arxiv else "pdf"
out_suffix = 'pdf'

count = 1

for line in file.readlines():
    emulateapj = emulateapjre.search(line)
    if line[0] == "%":
        continue
    elif emulateapj is not None:
        print "EmulateApJ -> aastex"
        print >>outf,"\\documentclass[12pt,preprint]{aastex}",
        f.close()
        continue
    fig  = figre.search(line)
    fig2 = fig2re.search(line)
    fig2b = fig2re1line.search(line)
    fig4 = fig4re.search(line)
    fig4b = fig4re1line.search(line)
    pone = plotonere.search(line)
    lonely = lonelygraphics.search(line)
    igr  = includegre.search(line)

    if igr is not None:
        fign = igr.groups()[0]
        prevline = ''
    elif fig is not None:
        fign = fig.groups()[0]
        prevline = ''
    elif fig2 is not None: 
        fign = fig2.groups()[0:2]
        prevline = ''
    elif fig4 is not None: 
        fign = fig4.groups()[0:4]
        prevline = ''
    elif fig4b is not None:
        fign = [n for n in fig4b.groups() if n is not None]
        nfigs = len(fign)
        prevline = 'fig4'
    elif fig2b is not None:
        fign = fig2b.groups()[0]
        nfigs = 1
        prevline = 'fig2'
    elif pone is not None: 
        fign = pone.groups()[0]
        prevline = ''
    elif lonely is not None and prevline in ('fig4','fig2'):
        print "in lonely: ",lonely.groups()
        fign = lonely.groups()[0]
        nfigs += 1
        if nfigs >= int(prevline[-1]):
            prevline = ''
    else:
        fign=None
        prevline = ''

    input = inputre.search(line)

    if fign is not None:
        #DEBUG print "Found fign: %s" % fign
        if fign in ("#1","#2","#3","#4"):
            print >>outf,line,
            continue
        if len(fign) == 1 or isinstance(fign,basestring):
            figlist = [fign]
        else:
            figlist = fign
        #DEBUG print "Figlist: ",figlist
        outline = line
        for fign in figlist:
            fignroot = fign
            if fign[-4] != '.': # if no suffix, add one
                fign += default_suffix
            if fign[-3:] == "png": 
                if args.reconvert:
                    os.system("pngtoeps %s" % fign)
                fn = ppath+fign.replace("png",out_suffix)
            elif fign[-3:] == "svg": 
                if args.reconvert:
                    os.system("svg2eps %s" % fign)
                fn = ppath+fign.replace("svg",out_suffix)
            elif fign[-3:] == "pdf": 
                if args.reconvert:
                    os.system("pdf2ps %s %s" % (fign,fign.replace("pdf","ps")))
                    os.system("mv %s %s" % (fign.replace('pdf','ps'),fign.replace('pdf','eps')))
                fn = ppath+fign.replace("pdf",out_suffix)
            elif fign[-3:] != out_suffix:
                fn = ppath+fign+"."+out_suffix
            elif fign[-3:] == "tex":
                raise TypeError("Figure is a text file? %s" % fign)
            else:
                fn = ppath+fign
            if os.system('ls %s' % fn) != 0:
                import pdb; pdb.set_trace()
            print "Converting figure " + fn + " to f%i.%s" % (count,out_suffix)
            outfig = 'f%i.%s' % (count,out_suffix)
            if os.system('cp %s %s' % (fn,ppath+"/"+outdir+"/"+outfig)) != 0:
                import pdb; pdb.set_trace()
            outline = outline.replace(fignroot,outfig)
            count += 1
        print >>outf,outline,
    elif input is not None:
        fn = input.groups()[0] + ".tex"
        print "Doing input " + fn
        f = open(ppath+fn,'r')
        if 'preface' in line:
            print >>outf,f.read(),
        else:
            print >>outf,strip_input(f.readlines()),
        f.close()
    else:
        print >>outf,line,

outf.close()
file.close()

os.chdir(ppath)
os.system('cp %s %s/ms.tex' % (outfn,outdir))
if os.path.exists(outdir+'/Makefile'):
    os.chdir(outdir)
    os.system('make')
    os.system('rm ms.dvi')
    os.system('rm ms.ps')
    os.system('rm ms.aux')
    os.system('rm ms.log')
    os.chdir(ppath)
    os.system('mv %s/ms.pdf %s_draft%s.pdf' % (outdir,paper_name,time.strftime("%m%d",time.localtime())))
os.system('tar --exclude Makefile -czf Ginsburg_H2COTurbulence_%s_%s.tar.gz %s/ ' % (time.strftime("%m%d",time.localtime()),outtype,outdir))



