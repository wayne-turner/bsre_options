import numpy as np
import matplotlib.pyplot as plt
from bsre_options import BSREOptions

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
