import numpy as np

class BSREOptions:
    """
    monte carlo engine for a real-estate style option w/ stochastic short rates & variance for prop value
    """
    def __init__(
        self,
        V0: float,
        K: float,
        T: float,
        r0: float,
        kappa_r: float,
        theta_r: float,
        sigma_r: float,
        v0: float,
        kappa_v: float,
        theta_v: float,
        sigma_v: float,
        rho: float,
        q: float,
        carry_cost: float = 0.0,
        M: int = 50,
        I: int = 10_000,
        seed: int | None = None,
    ) -> None:
        """
        V0 : float
            Initial real estate value at time 0.
        K : float
            Option strike.
        T : float
            Time to maturity in years.
        r0 : float
            Initial short rate.
        kappa_r, theta_r, sigma_r : float
            CIR mean reversion speed, long-run mean and volatility
            for the short rate.
        v0 : float
            Initial variance for the asset process.
        kappa_v, theta_v, sigma_v : float
            Mean reversion speed, long-run mean and volatility for
            the variance process.
        rho : float
            Correlation between the Brownian motions of V and r.
        q : float
            Non-negative continuous lease/dividend yield as a fraction
            of the property value per year.
        carry_cost : float
            Additional carry term added to the drift of V. Positive
            values increase the drift, negative values represent extra
            costs.
        M : int
            Number of time steps.
        I : int
            Number of simulated paths.
        seed : int or None
            seed for np.random.default_rng, if None, use entropy
        
        """
        self.V0 = float(V0)
        self.K = float(K)
        self.T = float(T)

        self.r0 = float(r0)
        self.kappa_r = float(kappa_r)
        self.theta_r = float(theta_r)
        self.sigma_r = float(sigma_r)

        self.v0 = float(v0)
        self.kappa_v = float(kappa_v)
        self.theta_v = float(theta_v)
        self.sigma_v = float(sigma_v)

        self.rho = float(rho)
        self.q = float(q)
        self.carry_cost = float(carry_cost)

        self.M = int(M)
        self.I = int(I)

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if self.q < 0.0:
            raise ValueError("q (lease/dividend yield) must be non-negative.")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError("rho must be in [-1, 1].")
        if self.M <= 0 or self.I <= 0:
            raise ValueError("M and I must be positive integers.")

    def simulate_paths(self) -> tuple[np.ndarray, np.ndarray]:
        """
        simulate joint paths for (V_t, r_t)

        returns:
        V : ndarray, shape (M+1, I)
            Simulated property values.
        r : ndarray, shape (M+1, I)
            Simulated short rates
        """
        dt = self.T / self.M
        sqrt_dt = np.sqrt(dt)

        V = np.empty((self.M + 1, self.I), dtype=float)
        r = np.empty((self.M + 1, self.I), dtype=float)
        v = np.empty((self.M + 1, self.I), dtype=float)

        V[0, :] = self.V0
        r[0, :] = self.r0
        v[0, :] = self.v0

        for t in range(1, self.M + 1):
            r_prev = r[t - 1]
            v_prev = v[t - 1]
            V_prev = V[t - 1]

            z_r = self.rng.standard_normal(self.I)
            z_s_indep = self.rng.standard_normal(self.I)
            z_v = self.rng.standard_normal(self.I)

            z_s = self.rho * z_r + np.sqrt(max(1.0 - self.rho**2, 0.0)) * z_s_indep

            r_drift = self.kappa_r * (self.theta_r - r_prev) * dt
            r_diff = self.sigma_r * np.sqrt(np.maximum(r_prev, 0.0)) * sqrt_dt * z_r
            r_t = r_prev + r_drift + r_diff
            r_t = np.maximum(r_t, 0.0)

            v_drift = self.kappa_v * (self.theta_v - v_prev) * dt
            v_diff = self.sigma_v * np.sqrt(np.maximum(v_prev, 0.0)) * sqrt_dt * z_v
            v_t = v_prev + v_drift + v_diff
            v_t = np.maximum(v_t, 0.0)

            drift = (r_prev + self.carry_cost - self.q - 0.5 * v_prev) * dt
            diff = np.sqrt(np.maximum(v_prev, 0.0)) * sqrt_dt * z_s
            V_t = V_prev * np.exp(drift + diff)

            V[t] = V_t
            r[t] = r_t
            v[t] = v_t

        return V, r

    def value_option(
        self,
        price_only: bool = True,
    ) -> float | tuple[float, float, tuple[float, float]]:
        """
        price a european call option on V_T by monte carlo
        uses the simulated paths, computes payoff, discounts w/ pathwise integral of short rate

        parameters:
        price_only : bool, default True
            If True, return only the Monte Carlo price as a scalar
            If False, return a tuple (price, stderr, (ci_lower, ci_upper))

        returns:
        float or tuple
            Monte Carlo price if price_only is True, otherwise
            (price, stderr, (ci_lower, ci_upper))
        """
        V, r = self.simulate_paths()
        payoff = np.maximum(V[-1] - self.K, 0.0)
        dt = self.T / self.M

        integral_r = np.sum(r[:-1], axis=0) * dt
        discount_factor = np.exp(-integral_r)

        price_paths = discount_factor * payoff
        price = float(np.mean(price_paths))

        if price_only:
            return price

        if self.I < 2:
            raise ValueError(
                "cannot compute monte carlo error, fewer than 2 paths, increase I or call value_option(price_only=True)"
            )

        sample_std = float(np.std(price_paths, ddof=1))
        stderr = sample_std / np.sqrt(self.I)

        z_95 = 1.96
        ci_lower = price - z_95 * stderr
        ci_upper = price + z_95 * stderr

        return price, stderr, (ci_lower, ci_upper)


def main(argv: list[str] | None = None) -> int:
    """
    minimal command-line interface for pricing a simple call option on the property value
    """
    import argparse

    parser = argparse.ArgumentParser(description="price a real-estate style option via monte carlo")
    parser.add_argument("--V0",type=float,default=200000.0,help="initial property value",)
    parser.add_argument("--K",type=float,default=220000.0,help="option strike",)
    parser.add_argument("--T",type=float,default=1.0,help="time to maturity in years",)
    parser.add_argument("--r0",type=float,default=0.03,help="initial short rate",)
    parser.add_argument("--kappa-r",type=float,default=0.3,help="CIR mean reversion speed for the short rate",)
    parser.add_argument("--theta-r",type=float,default=0.03,help="long-run mean level for the short rate",)
    parser.add_argument("--sigma-r",type=float,default=0.01,help="volatility of the short rate",)
    parser.add_argument("--v0",type=float,default=0.02,help="initial variance for the asset process",)
    parser.add_argument("--kappa-v",type=float,default=1.0,help="mean reversion speed for the variance process",)
    parser.add_argument("--theta-v",type=float,default=0.02,help="long-run mean variance level",)
    parser.add_argument("--sigma-v",type=float,default=0.1,help="volatility of the variance process",)
    parser.add_argument("--rho",type=float,default=-0.5,help="correlation between asset and rate shocks",)
    parser.add_argument("--q",type=float,default=0.02,help="continuous lease/dividend yield",)
    parser.add_argument("--carry-cost",type=float,default=0.0,help="additional carry term in the asset drift (can be negative)",)
    parser.add_argument("--M",type=int,default=100,help="number of time steps for the simulation",)
    parser.add_argument("--I",type=int,default=10_000,help="number of monte carlo paths",)
    parser.add_argument("--seed",type=int,default=123,help="random seed for reproducible simulations",)
    args = parser.parse_args(argv)

    model = BSREOptions(
        V0=args.V0,
        K=args.K,
        T=args.T,
        r0=args.r0,
        kappa_r=args.kappa_r,
        theta_r=args.theta_r,
        sigma_r=args.sigma_r,
        v0=args.v0,
        kappa_v=args.kappa_v,
        theta_v=args.theta_v,
        sigma_v=args.sigma_v,
        rho=args.rho,
        q=args.q,
        carry_cost=args.carry_cost,
        M=args.M,
        I=args.I,
        seed=args.seed,
    )
    price = model.value_option()
    print(f"estimated option value: {price:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
