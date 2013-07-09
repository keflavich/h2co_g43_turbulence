import pymc

def make_ratio_distribution(value1, error1, value2, error2):
    tau1 = 1./error1**2
    tau2 = 1./error2**2
    d = {}
    d['numerator'] = pymc.Normal('numerator', value1, tau1)
    d['denominator'] = pymc.Normal('denominator', value2, tau2)

    @pymc.deterministic(trace=True)
    def ratio(numerator=d['numerator'], denominator=d['denominator']):
        return numerator/denominator

    d['ratio'] = ratio

    return pymc.MCMC(d)
