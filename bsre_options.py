import numpy as np

class BSREOptions:
    
    def __init__(self, V0, K, T, r0, kappa_r, theta_r, sigma_r, v0, kappa_v, theta_v, sigma_v, rho, lease_income, M, I):
        self.V0, self.K, self.T, self.lease_income = V0, K, T, lease_income
        self.r0 = r0
        self.kappa_r, self.theta_r, self.sigma_r = kappa_r, theta_r, sigma_r
        self.v0, self.kappa_v, self.theta_v, self.sigma_v = v0, kappa_v, theta_v, sigma_v
        self.rho = rho
        self.M, self.I = M, I
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

    def simulate_paths(self):
        """
        simulates RE value paths under stochastic conditions, reflecting impact of interest rates, and prop value volatility over time
        """
        dt = self.T / self.M
        V = np.zeros((self.M + 1, self.I))
        r = np.zeros_like(V)
        v = np.zeros_like(V)
        V[0] = self.V0
        r[0] = self.r0
        v[0] = self.v0

        for t in range(1, self.M + 1):
            z1 = np.random.standard_normal(self.I)
            z2 = np.random.standard_normal(self.I)
            z3 = np.random.standard_normal(self.I)
            z_r = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            # update interest using CIR
            r[t] = r[t-1] + self.kappa_r * (self.theta_r - r[t-1]) * dt + self.sigma_r * np.sqrt(np.maximum(r[t-1], 0) * dt) * z_r
            r[t] = np.maximum(r[t], 0)

            # update volatility
            v[t] = v[t-1] + self.kappa_v * (self.theta_v - v[t-1]) * dt + self.sigma_v * np.sqrt(np.maximum(v[t-1], 0) * dt) * z3
            v[t] = np.maximum(v[t], 0)

            # dividend yield from leaseincome
            dividend_yield = self.lease_income / V[t-1]

            # update asset price
            V[t] = V[t-1] * np.exp((r[t-1] - dividend_yield - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1] * dt) * z1)

        return V, r

    def value_option(self):
        """
        values lease option using the paths, calculating the expected payoff, discounting present value

        """
        V, r = self.simulate_paths()
        payoff = np.maximum(V[-1] - self.K, 0)  # adjust put/call
        dt = self.T / self.M
        discount_factor = np.exp(-np.sum(r, axis=0) * dt)
        C = discount_factor * payoff
        return np.mean(C)
