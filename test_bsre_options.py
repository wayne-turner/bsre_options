import unittest
import numpy as np
from bsre_options import BSREOptions

# initialization
class TestBSREOptionsInitialization(unittest.TestCase):
    def test_initialization(self):
        model = BSREOptions(
            V0=200000,
            K=220000,
            T=1,
            r0=0.03,
            kappa_r=0.3,
            theta_r=0.03,
            sigma_r=0.01,
            v0=0.02,
            kappa_v=1.0,
            theta_v=0.02,
            sigma_v=0.1,
            rho=-0.5,
            lease_income=-10000,
            M=100,
            I=10000
        )
        self.assertEqual(model.V0, 200000)
        self.assertEqual(model.K, 220000)
        self.assertEqual(model.T, 1)
        self.assertEqual(model.r0, 0.03)
        self.assertEqual(model.kappa_r, 0.3)
        self.assertEqual(model.theta_r, 0.03)
        self.assertEqual(model.sigma_r, 0.01)
        self.assertEqual(model.v0, 0.02)
        self.assertEqual(model.kappa_v, 1.0)
        self.assertEqual(model.theta_v, 0.02)
        self.assertEqual(model.sigma_v, 0.1)
        self.assertEqual(model.rho, -0.5)
        self.assertEqual(model.lease_income, -10000)
        self.assertEqual(model.M, 100)
        self.assertEqual(model.I, 10000)

# simulate paths
class TestBSREOptionsSimulation(unittest.TestCase):
    def setUp(self):
        self.model = BSREOptions(
            V0=200000,
            K=220000,
            T=1,
            r0=0.03,
            kappa_r=0.3,
            theta_r=0.03,
            sigma_r=0.01,
            v0=0.02,
            kappa_v=1.0,
            theta_v=0.02,
            sigma_v=0.1,
            rho=-0.5,
            lease_income=-10000,
            M=100,
            I=1000 
        )

    def test_simulate_paths_output_shape(self):
        V, r = self.model.simulate_paths()
        self.assertEqual(V.shape, (self.model.M + 1, self.model.I))
        self.assertEqual(r.shape, (self.model.M + 1, self.model.I))

    def test_simulate_paths_values(self):
        V, r = self.model.simulate_paths()
        # values non-negative
        self.assertTrue(np.all(V >= 0))
        self.assertTrue(np.all(r >= 0))
        # values correct
        self.assertTrue(np.all(V[0] == self.model.V0))
        self.assertTrue(np.all(r[0] == self.model.r0))

# option valuation
class TestBSREOptionsValuation(unittest.TestCase):
    def setUp(self):
        self.model = BSREOptions(
            V0=200000,
            K=220000,
            T=1,
            r0=0.03,
            kappa_r=0.3,
            theta_r=0.03,
            sigma_r=0.01,
            v0=0.02,
            kappa_v=1.0,
            theta_v=0.02,
            sigma_v=0.1,
            rho=-0.5,
            lease_income=-10000,
            M=100,
            I=1000
        )

    def test_value_option_returns_float(self):
        option_price = self.model.value_option()
        self.assertIsInstance(option_price, float)

    def test_value_option_positive_price(self):
        option_price = self.model.value_option()
        self.assertGreaterEqual(option_price, 0)

    def test_zero_volatility(self):
        self.model.v0 = 0
        self.model.sigma_v = 0
        self.model.kappa_v = 0
        self.model.theta_v = 0 
        self.model.sigma_r = 0
        self.model.kappa_r = 0
        self.model.theta_r = self.model.r0

        # dividend yield
        dividend_yield = self.model.lease_income / self.model.V0
        # drift
        mu = self.model.r0 - dividend_yield
        # price at maturity
        V_T = self.model.V0 * np.exp(mu * self.model.T)
        # payoff
        payoff = max(V_T - self.model.K, 0)
        # payof fpresent value
        expected_price = payoff * np.exp(-self.model.r0 * self.model.T)
        # price option
        option_price = self.model.value_option()
        # price ~ expected price
        self.assertAlmostEqual(option_price, expected_price, delta=1e-2)

# extreme parameters
class TestBSREOptionsExtremeParameters(unittest.TestCase):
    def test_high_volatility(self):
        model = BSREOptions(
            V0=200000,
            K=220000,
            T=1,
            r0=0.03,
            kappa_r=0.3,
            theta_r=0.03,
            sigma_r=0.01,
            v0=1.0,  # high volatility
            kappa_v=1.0,
            theta_v=1.0,
            sigma_v=0.5,
            rho=-0.5,
            lease_income=-10000,
            M=100,
            I=1000
        )
        option_price = model.value_option()
        self.assertGreater(option_price, 0)

    def test_negative_correlation(self):
        model = BSREOptions(
            V0=200000,
            K=220000,
            T=1,
            r0=0.03,
            kappa_r=0.3,
            theta_r=0.03,
            sigma_r=0.01,
            v0=0.02,
            kappa_v=1.0,
            theta_v=0.02,
            sigma_v=0.1,
            rho=-0.99,  # negative corr
            lease_income=-10000,
            M=100,
            I=1000
        )
        option_price = model.value_option()
        self.assertGreaterEqual(option_price, 0)
