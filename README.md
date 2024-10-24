# BSREOptions: Black-Scholes Real Estate Options

BSREOptions offers an adaptation of the traditional Black-Scholes model, tailored specifically for the unique dynamics of real estate options. This repository presents an implementation designed to evaluate lease options in real estate, integrating stochastic interest rates and property value volatility. BSREOptions is fundamentally an educational tool.

## Applications

The BSREOptions model finds its utility across a spectrum of real estate and financial scenarios:

- **RE Development and Investment**: Aiding developers and investors in assessing lease options.
- **Portfolio Management**: Offering REITs a tool for valuing lease options across diversified portfolios.
- **Lease Negotiation**: Equipping landlords and tenants with data to negotiate lease terms effectively.
- **Financing and Lending**: Evaluating the risk and potential return on RE projects involving lease options.
- **Insurance and Risk**: Helping insurers price products related to real estate investment risks accurately.

## Black-Scholes Shortcomings in RE

- RE markets have characteristics that can make the assumptions behind the Black-Scholes model (e.g., continuous trading, log-normal price distribution) less applicable.
- These assets are illiquid, have longer investment horizons, and their market values are influenced by a wide range of economic and local factors that can cause deviations from log-normal behavior.
- BSREOptions addresses some of these limitations by incorporating specific real estate market dynamics, offering a more accurate tool for evaluating lease options.

## How Others Can Use This Repository

This repository is intended for developers, financial analysts, real estate investors, and anyone interested in the quantitative analysis of real estate options. Users can clone the repository, install dependencies, and utilize the BSREOptions class to perform their analyses, adapting the model parameters to fit their specific needs. 


## Parameters and Reference Table

| Argument        | Type    | Description                                                                                      | Typical Ranges          |
|-----------------|---------|--------------------------------------------------------------------------------------------------|------------------------|
| `V0`            | float   | Initial real estate value, representing the market value of the property at the start of analysis. | $100,000 - $10,000,000 |
| `K`             | float   | Option strike price, the agreed-upon price for the option to buy or sell the property.             | $100,000 - $10,000,000 |
| `T`             | float   | Time to maturity of the lease option, measured in years.                                          | 1 - 10 years           |
| `r0`            | float   | Initial interest rate, representing the starting level of interest rates for the simulation.      | 0% - 10%               |
| `kappa_r`       | float   | Speed of adjustment to the mean for interest rate, determining how quickly rates revert to mean.   | 0.1 - 3                |
| `theta_r`       | float   | Long-term mean of interest rate, the average level to which the interest rate reverts over time.   | 0% - 10%               |
| `sigma_r`       | float   | Volatility of interest rate, indicating the degree of variation in interest rates from the mean.   | 0.01 - 0.5             |
| `v0`            | float   | Initial volatility of real estate value, representing the starting variability in property values. | 0.01 - 0.5             |
| `kappa_v`       | float   | Speed of adjustment to the mean for volatility, how quickly volatility reverts to its long-term mean. | 0.1 - 3              |
| `theta_v`       | float   | Long-term mean of volatility, the average level of volatility to which the property value's volatility reverts. | 0.01 - 0.5       |
| `sigma_v`       | float   | Volatility of volatility, indicating the degree of variation in the volatility from the mean.      | 0.01 - 0.5             |
| `rho`           | float   | Correlation between property value and interest rate, measuring how these two factors move in relation to each other. | -1 to 1 |
| `lease_income`  | float   | Annual net lease income, the net income received from leasing the property, negative for costs.    | -$100,000 - $1,000,000 |
| `M`             | int     | Number of time steps for the simulation, determining the granularity of the valuation timeline.    | 50 - 1000              |
| `I`             | int     | Number of simulation paths, affecting the model's accuracy and computational intensity.             | 1000 - 10000           |





## Quickstart

This guide provides a quick overview of how to use the `BSREOptions` class to simulate real estate value paths under specific financial conditions and visualize the results. The class allows for the simulation of the evolution of real estate values over time, considering interest rate changes and lease income adjustments.

### Initialization

```python
from bsre_options import BSREOptions

model = BSREOptions(
    V0=200000,  # Initial real estate value
    K=220000,  # Option strike price 
    T=1,  # Time to maturity
    r0=0.03,  # Initial interest rate
    kappa_r=0.3,  # Rate of mean reversion for interest rate
    theta_r=0.03,  # Long-term interest rate mean
    sigma_r=0.01,  # Interest rate volatility
    v0=0.02,  # Volatility of real estate value
    kappa_v=1.0,  # Rate of mean reversion for volatility
    theta_v=0.02,  # Long-term volatility mean
    sigma_v=0.1,  # Volatility of volatility
    rho=-0.5,  # Correlation between the asset value and interest rate
    lease_income=-10000,  # Annual lease income
    M=100,  # Number of time steps
    I=10000  # Number of simulation paths
)
```


### Estimate Value
```python
option_price = model.value_option()
print(f"Estimated Lease Option Value: {option_price:.2f}")
```

### Generate Plot
```python
V, r = model.simulate_paths()

# simulate real estate value paths
plt.figure(figsize=(10, 6), facecolor='#121212')
for i in range(min(100, model.I)):
    plt.plot(V[:, i], alpha=0.5, color='gray')
plt.title('Simulated Real Estate Value Paths', color='white')
plt.xlabel('Time Steps', color='white')
plt.ylabel('Real Estate Value', color='white')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
plt.show()
```







## Simulated Real Estate Value Paths

Visualization helps in understanding how real estate values might change over time under different scenarios, providing insights into the variability and risk associated with real estate investments. Each gray line represents a possible trajectory of real estate value, starting from an initial value and evolving according to specified market dynamics and random fluctuations.

<img src="assets/re_value_paths.png" width="70%" />

## Distribution of Final Real Estate Values

The distribution of final real estate values across all simulated scenarios at the end of the specified time period. This gauges the spread of possible end values, highlighting the risks and opportunities in real estate investments based on the model's assumptions and parameters.

<img src="assets/re_value_dist.png" width="70%" />









## Output Interpretation

The output of the BSREOptions model is the estimated value of a lease option, expressed in USD. This estimate is the result of complex simulations that account for various factors, including interest rates, property value volatility, lease income, and more. Here's how to interpret this result:

- **Estimated Lease Option Value**: This is the central output of the model, representing the present value of the expected payoff from exercising the lease option under simulated market conditions. A higher value suggests that, under the parameters provided, the lease option could be financially beneficial if the conditions reflect real-world scenarios accurately.

- **Understanding the Value**:
  - If the option value is significantly higher than zero, it suggests that the option to lease (or buy/sell at the strike price) is valuable under the given market conditions and parameters.
  - A value close to zero indicates that the lease option might not provide substantial financial benefit, or the option's premium might be close to its fair market value.
  - It's crucial to remember that this value is contingent upon the accuracy of the model's parameters and assumptions. Real-world factors will lead to different outcomes.



## Conclusion
- **Educational Aim**: BSREOptions is fundamentally an educational tool, designed to deepen understanding of real estate options.
- **Adaptability**: Encourages adaptation and exploration, inviting users to tailor the model to their specific contexts.
- **Starting Point**: Acts as a springboard for further research and innovation in the field of real estate financial analysis.


