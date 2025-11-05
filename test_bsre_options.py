import unittest
import numpy as np
from bsre_options import BSREOptions, main

class TestBSREOptionsInitialization(unittest.TestCase):
    def test_initialization(self) -> None:
        model = BSREOptions(
            V0=200000.0,
            K=220000.0,
            T=1.0,
            r0=0.03,
            kappa_r=0.3,
            theta_r=0.03,
            sigma_r=0.01,
            v0=0.02,
            kappa_v=1.0,
            theta_v=0.02,
            sigma_v=0.1,
            rho=-0.5,
            q=0.05,
            carry_cost=-0.01,
            M=100,
            I=1000,
            seed=42,
        )

        self.assertEqual(model.V0, 200000.0)
        self.assertEqual(model.K, 220000.0)
        self.assertEqual(model.T, 1.0)
        self.assertEqual(model.r0, 0.03)
        self.assertEqual(model.kappa_r, 0.3)
        self.assertEqual(model.theta_r, 0.03)
        self.assertEqual(model.sigma_r, 0.01)
        self.assertEqual(model.v0, 0.02)
        self.assertEqual(model.kappa_v, 1.0)
        self.assertEqual(model.theta_v, 0.02)
        self.assertEqual(model.sigma_v, 0.1)
        self.assertEqual(model.rho, -0.5)
        self.assertEqual(model.q, 0.05)
        self.assertEqual(model.carry_cost, -0.01)
        self.assertEqual(model.M, 100)
        self.assertEqual(model.I, 1000)

    def test_invalid_q_raises(self) -> None:
        with self.assertRaises(ValueError):
            BSREOptions(
                V0=1.0,
                K=1.0,
                T=1.0,
                r0=0.01,
                kappa_r=0.1,
                theta_r=0.01,
                sigma_r=0.01,
                v0=0.01,
                kappa_v=0.1,
                theta_v=0.01,
                sigma_v=0.1,
                rho=0.0,
                q=-0.01,
                carry_cost=0.0,
                M=10,
                I=10,
            )


class TestBSREOptionsSimulation(unittest.TestCase):
    def _make_model(self, seed: int | None = 123) -> BSREOptions:
        return BSREOptions(
            V0=200000.0,
            K=220000.0,
            T=1.0,
            r0=0.03,
            kappa_r=0.3,
            theta_r=0.03,
            sigma_r=0.01,
            v0=0.02,
            kappa_v=1.0,
            theta_v=0.02,
            sigma_v=0.1,
            rho=-0.5,
            q=0.02,
            carry_cost=0.0,
            M=50,
            I=1000,
            seed=seed,
        )

    def test_simulate_paths_shape_and_start_values(self) -> None:
        model = self._make_model()
        V, r = model.simulate_paths()
        self.assertEqual(V.shape, (model.M + 1, model.I))
        self.assertEqual(r.shape, (model.M + 1, model.I))
        np.testing.assert_allclose(V[0], model.V0)
        np.testing.assert_allclose(r[0], model.r0)

    def test_simulate_paths_no_nans(self) -> None:
        model = self._make_model()
        V, r = model.simulate_paths()
        self.assertFalse(np.isnan(V).any())
        self.assertFalse(np.isnan(r).any())


class TestBSREOptionsDeterministicZeroVolatility(unittest.TestCase):
    def test_deterministic_zero_vol(self) -> None:
        V0 = 100.0
        K = 90.0
        T = 1.0
        r0 = 0.05
        q = 0.02
        carry_cost = -0.01

        model = BSREOptions(
            V0=V0,
            K=K,
            T=T,
            r0=r0,
            kappa_r=0.0,
            theta_r=r0,
            sigma_r=0.0,
            v0=0.0,
            kappa_v=0.0,
            theta_v=0.0,
            sigma_v=0.0,
            rho=0.0,
            q=q,
            carry_cost=carry_cost,
            M=50,
            I=10,
            seed=7,
        )

        drift = r0 + carry_cost - q
        V_T = V0 * np.exp(drift * T)
        payoff = max(V_T - K, 0.0)
        expected_price = payoff * np.exp(-r0 * T)

        option_price = model.value_option()
        self.assertAlmostEqual(option_price, expected_price, places=6)

        price, stderr, (ci_low, ci_high) = model.value_option(price_only=False)
        self.assertAlmostEqual(price, expected_price, places=6)
        self.assertAlmostEqual(stderr, 0.0, places=12)
        self.assertAlmostEqual(ci_low, expected_price, places=6)
        self.assertAlmostEqual(ci_high, expected_price, places=6)


class TestBSREOptionsSeeding(unittest.TestCase):
    def _make_model(self, seed: int | None) -> BSREOptions:
        return BSREOptions(
            V0=200000.0,
            K=220000.0,
            T=1.0,
            r0=0.03,
            kappa_r=0.3,
            theta_r=0.03,
            sigma_r=0.01,
            v0=0.02,
            kappa_v=1.0,
            theta_v=0.02,
            sigma_v=0.1,
            rho=-0.5,
            q=0.05,
            carry_cost=0.0,
            M=50,
            I=1000,
            seed=seed,
        )

    def test_paths_reproducible_with_same_seed(self) -> None:
        model1 = self._make_model(seed=123)
        model2 = self._make_model(seed=123)

        V1, r1 = model1.simulate_paths()
        V2, r2 = model2.simulate_paths()

        np.testing.assert_allclose(V1, V2)
        np.testing.assert_allclose(r1, r2)

    def test_paths_differ_with_different_seeds(self) -> None:
        model1 = self._make_model(seed=123)
        model2 = self._make_model(seed=456)

        V1, r1 = model1.simulate_paths()
        V2, r2 = model2.simulate_paths()

        self.assertFalse(np.allclose(V1, V2))
        self.assertFalse(np.allclose(r1, r2))

    def test_value_option_reproducible_with_same_seed(self) -> None:
        model1 = self._make_model(seed=123)
        model2 = self._make_model(seed=123)

        price1 = model1.value_option()
        price2 = model2.value_option()

        self.assertAlmostEqual(price1, price2, places=10)

    def test_value_option_price_only_backward_compatible(self) -> None:
        model1 = self._make_model(seed=123)
        model2 = self._make_model(seed=123)

        price_default = model1.value_option()
        price_price_only = model2.value_option(price_only=True)

        self.assertIsInstance(price_default, float)
        self.assertIsInstance(price_price_only, float)
        self.assertAlmostEqual(price_default, price_price_only, places=10)


class TestBSREOptionsMonteCarloError(unittest.TestCase):
    def test_error_if_insufficient_paths(self) -> None:
        model = BSREOptions(
            V0=1.0,
            K=1.0,
            T=1.0,
            r0=0.01,
            kappa_r=0.1,
            theta_r=0.01,
            sigma_r=0.01,
            v0=0.01,
            kappa_v=0.1,
            theta_v=0.01,
            sigma_v=0.1,
            rho=0.0,
            q=0.01,
            carry_cost=0.0,
            M=10,
            I=1,
            seed=1,
        )

        with self.assertRaises(ValueError):
            model.value_option(price_only=False)


class TestCLI(unittest.TestCase):
    def test_main_runs_with_small_benchmark_size(self) -> None:
        exit_code = main(["--I", "100", "--M", "10"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
