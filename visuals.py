import numpy as np
import matplotlib.pyplot as plt

class BSREOptions:
    def __init__(self, V0, K, T, r0, kappa_r, theta_r, sigma_r, v0, kappa_v, theta_v, sigma_v, rho, lease_income, M, I):
        self.V0, self.K, self.T, self.lease_income = V0, K, T, lease_income
        self.r0 = r0
        self.kappa_r, self.theta_r, self.sigma_r = kappa_r, theta_r, sigma_r
        self.v0, self.kappa_v, self.theta_v, self.sigma_v, self.rho = v0, kappa_v, theta_v, sigma_v, rho
        self.M, self.I = M, I

    def simulate_paths(self):
        dt = self.T / self.M
        V = np.zeros((self.M + 1, self.I))
        r = np.zeros_like(V)
        V[0] = self.V0
        r[0] = self.r0
        
        for t in range(1, self.M + 1):
            z1 = np.random.standard_normal(self.I)
            z2 = np.random.standard_normal(self.I)
            r[t] = np.maximum(r[t-1] + self.kappa_r * (self.theta_r - r[t-1]) * dt + self.sigma_r * np.sqrt(dt) * z2, 0)
            adjusted_lease_income = self.lease_income * np.exp(-r[t-1] * dt)
            V[t] = V[t-1] * np.exp((adjusted_lease_income / V[t-1] - 0.5 * self.v0) * dt + np.sqrt(self.v0 * dt) * z1)

        return V, r

# initialize
model = BSREOptions(
    V0=200000, K=220000, T=1, r0=0.03, kappa_r=0.3, theta_r=0.03, sigma_r=0.01,
    v0=0.02, kappa_v=1.0, theta_v=0.02, sigma_v=0.1, rho=-0.5, lease_income=-10000, M=100, I=10000
)

# gen sim paths
V, r = model.simulate_paths()

# monte
plt.figure(figsize=(10, 6), facecolor='#121212')
for i in range(min(100, model.I)):
    plt.plot(V[:, i], alpha=0.5, color='gray')
plt.title('Simulated Real Estate Value Paths', color='white')
plt.xlabel('Time Steps', color='white')
plt.ylabel('Real Estate Value', color='white')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
plt.show()

# hist
plt.figure(figsize=(10, 6), facecolor='#121212')
plt.hist(V[-1, :], bins=50, alpha=0.75, color='gray')
plt.title('Distribution of Final Real Estate Values', color='white')
plt.xlabel('Final Real Estate Value', color='white')
plt.ylabel('Frequency', color='white')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
plt.show()

# heatmap
plt.figure(figsize=(10, 6), facecolor='#121212') 
heatmap, xedges, yedges = np.histogram2d(r.flatten(), V.flatten(), bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='gray')
plt.title('Heatmap of Interest Rates and Real Estate Values', color='white')
plt.xlabel('Interest Rate', color='white')
plt.ylabel('Real Estate Value', color='white')
plt.colorbar(label='Frequency').ax.yaxis.set_tick_params(color='white')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
plt.show()
