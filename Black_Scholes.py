import numpy as np
import scipy.stats as stats

class BlackScholes:
    def __init__(self,K,S,t,sigma,r=0.05):
        self.K = K
        self.S = S
        self.t = t/365
        self.sigma = sigma
        self.r = r

    def Black_Scholes_Call_Option(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)
        C = stats.norm.cdf(d1, 0.0, 1) * self.S - stats.norm.cdf(d2, 0.0, 1) * self.K * np.exp(-self.r * self.t)
        return format(C, '.4f')

    def Black_Scholes_Put_Option(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)
        P = stats.norm.cdf(-d2, 0.0, 1) * self.K * np.exp(-self.r * self.t) - stats.novrm.cdf(-d1, 0.0, 1) * self.S
        return format(P, '.4f')

bsm = BlackScholes(100,95,365,0.2)
print(bsm.Black_Scholes_Call_Option())