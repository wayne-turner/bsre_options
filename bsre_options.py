import numpy as np

class BSREOptions:
    def __init__(self, V0, K, T, r0, kappa_r, theta_r, sigma_r, v0, kappa_v, theta_v, sigma_v, rho, lease_income, M, I):
        """
        initializes model
        
        parameters:
        - V0 (float): initial RE value
        - K (float): option strike price
        - T (float): time to maturity of  lease option
        - r0 (float): initial int rate
        - kappa_r (float):speed of adj to mean interest rate
        - theta_r (float): long-term mean interest rate
        - sigma_r (float):volatility of interest rate
        - v0 (float): initial volatility of RE value
        - kappa_v (float): speed of adj to mean volatility
        - theta_v (float): long-term mean volatility
        - sigma_v (float): volatility of volatility
        - rho (float): correlation prop value and interest rate
        - lease_income (float): annual net lease income
        - M (int): # of time steps for the simulation
        - I (int): # of simulation paths
        """
        self.V0, self.K, self.T, self.lease_income = V0, K, T, lease_income
        self.r0 = r0
        self.kappa_r, self.theta_r, self.sigma_r = kappa_r, theta_r, sigma_r
        self.v0, self.kappa_v, self.theta_v, self.sigma_v, self.rho = v0, kappa_v, theta_v, sigma_v, rho
        self.M, self.I = M, I

    def simulate_paths(self):
        """
        simulates RE value paths under stochastic conditions, reflecting impact of interest rates,
        and prop value volatility over time
      
        returns:
        - V (numpy.ndarray): simulated RE value paths
        - r (numpy.ndarray): simulated interest rate paths
        """
        dt = self.T / self.M
        V = np.zeros((self.M + 1, self.I))
        r = np.zeros_like(V)
        V[0] = self.V0
        r[0] = self.r0
        
        for t in range(1, self.M + 1):
            z1 = np.random.standard_normal(self.I)
            z2 = np.random.standard_normal(self.I)
            z3 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            r[t] = np.maximum(r[t-1] + self.kappa_r * (self.theta_r - r[t-1]) * dt + self.sigma_r * np.sqrt(dt) * z2, 0)
            adjusted_lease_income = self.lease_income * np.exp(-r[t-1] * dt)
            V[t] = V[t-1] * np.exp((adjusted_lease_income / V[t-1] - 0.5 * self.v0) * dt + np.sqrt(self.v0 * dt) * z1)

        return V, r

    def value_option(self):
        """
        values lease option using the paths, calculating the expected payoff, discounting present value
        
        returns:
        - (float): est value of the lease option
        """
        V, r = self.simulate_paths()
        payoff = np.maximum(V[-1] - self.K, 0)  # call/put adjust
        C = np.exp(-r.sum(axis=0) * self.T / self.M) * payoff
        return np.mean(C)
