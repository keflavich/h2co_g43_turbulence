import pymc
import numpy as np
import astropy.table as table

def stats_table(mc,quantiles=(0.1,1,2.5, 25, 50, 75, 97.5, 99, 99.9)):
    stats = mc.stats(quantiles=quantiles)
    keys = ['n', 'standard deviation', 'mc error', 'mean']
    stats_keys = keys+[]  # copy that will not include the others...
    keys += ['95%HPDInterval_'+x for x in ('lo','hi')]
    keys += ['q%0.1f' % q for q in quantiles]
    dtypes = [np.str]+[np.float] * len(keys)

    rows = {}
    for k in stats:
        rows[k] = [k]+[stats[k][s] for s in stats_keys]
        rows[k] += stats[k]['95% HPD interval'].tolist()
        rows[k] += [stats[k]['quantiles'][q] for q in quantiles]

    data = [rows[k] for k in rows]
    data = map(list,zip(*data))

    names = ['variable name'] + keys

    T = table.Table(data=data,names=names,dtypes=dtypes)
    return T
