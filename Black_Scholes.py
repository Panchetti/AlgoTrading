import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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

class Monte_Carlo:
    def __init__(self,K,S_0,t,sigma,simulations,r=0.05):
        self.K = K
        self.S_0 = S_0
        self.t = t/365
        self.sigma = sigma
        self.r = r

        self.N = simulations
        self.num_steps = t
        self.dt = self.t/self.num_steps
        self.simulation_results_S=None

    def simulate_prices(self):

        np.random.seed(20)

        S = np.zeros((self.num_steps,self.N))
        S[0,:]=self.S_0


        for t in range(1,self.num_steps):
            Z = np.random.standard_normal(self.N)
            S[t] = S[t-1]*np.exp((self.r - 0.5*self.sigma**2)*self.dt  + (self.sigma*np.sqrt(self.dt))*Z)

        self.simulation_results_S = S

    def simulate_call(self):
        if self.simulation_results_S is None:
            return -1
        else:
            return np.exp(-self.r*self.t)*1/self.N*np.sum(np.maximum(self.simulation_results_S[-1]-self.K,0))

    def simulate_put(self):
        if self.simulation_results_S is None:
            return -1
        else:
            return np.exp(-self.r*self.t)*1/self.N*np.sum(np.maximum(self.K-self.simulation_results_S[-1],0))

    def plot_simulation_results(self, num_of_movements):
        """Plots specified number of simulated price movements."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.simulation_results_S[:, 0:num_of_movements])
        plt.axhline(self.K, c='k', xmin=0, xmax=self.num_steps, label='Strike Price')
        plt.xlim([0, self.num_steps])
        plt.ylabel('Simulated price movements')
        plt.xlabel('Days in future')
        plt.title(f'First {num_of_movements}/{self.N} Random Price Movements')
        plt.legend(loc='best')
        plt.show()
mc = Monte_Carlo(100,100,30,0.2,simulations=1000)
mc.simulate_prices()
print(mc.simulate_call())
print(mc.simulate_put())
print(mc.plot_simulation_results(num_of_movements=1000))

