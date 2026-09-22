"""
options_pricer.py

Object-oriented vanilla options pricer.
Implements Black-Scholes (closed-form) and Monte Carlo pricing,
plus the main Greeks (delta, gamma, vega, theta, rho).

Author: (your name)
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.stats import norm


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class Option:
    """
    Represents a vanilla European option.

    Parameters
    ----------
    S0 : float   Spot price of the underlying
    K  : float   Strike price
    T  : float   Time to maturity, in years
    r  : float   Risk-free interest rate (annualized, continuous compounding)
    sigma : float   Volatility of the underlying (annualized)
    option_type : OptionType   CALL or PUT
    q : float   Continuous dividend yield (default 0)
    """

    S0: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType = OptionType.CALL
    q: float = 0.0

    # ---------- Black-Scholes ----------

    def _d1_d2(self):
        d1 = (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price_bs(self) -> float:
        """Closed-form Black-Scholes price."""
        d1, d2 = self._d1_d2()
        disc_r = np.exp(-self.r * self.T)
        disc_q = np.exp(-self.q * self.T)

        if self.option_type == OptionType.CALL:
            price = self.S0 * disc_q * norm.cdf(d1) - self.K * disc_r * norm.cdf(d2)
        else:
            price = self.K * disc_r * norm.cdf(-d2) - self.S0 * disc_q * norm.cdf(-d1)
        return float(price)

    # ---------- Greeks ----------

    def delta(self) -> float:
        d1, _ = self._d1_d2()
        disc_q = np.exp(-self.q * self.T)
        if self.option_type == OptionType.CALL:
            return float(disc_q * norm.cdf(d1))
        return float(disc_q * (norm.cdf(d1) - 1))

    def gamma(self) -> float:
        d1, _ = self._d1_d2()
        disc_q = np.exp(-self.q * self.T)
        return float(disc_q * norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T)))

    def vega(self) -> float:
        """Sensitivity to a 1.0 (100%) change in volatility. Divide by 100 for a 1% move."""
        d1, _ = self._d1_d2()
        disc_q = np.exp(-self.q * self.T)
        return float(self.S0 * disc_q * norm.pdf(d1) * np.sqrt(self.T))

    def theta(self) -> float:
        """Time decay, per year. Divide by 365 for a daily figure."""
        d1, d2 = self._d1_d2()
        disc_r = np.exp(-self.r * self.T)
        disc_q = np.exp(-self.q * self.T)
        term1 = -(self.S0 * disc_q * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))

        if self.option_type == OptionType.CALL:
            term2 = self.q * self.S0 * disc_q * norm.cdf(d1)
            term3 = -self.r * self.K * disc_r * norm.cdf(d2)
        else:
            term2 = -self.q * self.S0 * disc_q * norm.cdf(-d1)
            term3 = self.r * self.K * disc_r * norm.cdf(-d2)

        return float(term1 + term2 + term3)

    def rho(self) -> float:
        """Sensitivity to a 1.0 (100%) change in the risk-free rate. Divide by 100 for a 1% move."""
        _, d2 = self._d1_d2()
        disc_r = np.exp(-self.r * self.T)
        if self.option_type == OptionType.CALL:
            return float(self.K * self.T * disc_r * norm.cdf(d2))
        return float(-self.K * self.T * disc_r * norm.cdf(-d2))

    def greeks(self) -> dict:
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(),
            "rho": self.rho(),
        }

    # ---------- Monte Carlo ----------

    def price_mc(self, n_sims: int = 100_000, n_steps: int = 1, seed: int | None = None) -> tuple[float, float]:
        """
        Monte Carlo price under geometric Brownian motion (risk-neutral measure).

        Returns
        -------
        (price, standard_error)
        """
        rng = np.random.default_rng(seed)
        dt = self.T / n_steps

        # Simulate terminal log-price directly (exact GBM discretization)
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * rng.standard_normal(n_sims)
        S_T = self.S0 * np.exp(drift + diffusion)

        if self.option_type == OptionType.CALL:
            payoffs = np.maximum(S_T - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - S_T, 0.0)

        discounted = np.exp(-self.r * self.T) * payoffs
        price = discounted.mean()
        std_error = discounted.std(ddof=1) / np.sqrt(n_sims)
        return float(price), float(std_error)


if __name__ == "__main__":
    # Quick sanity check: BS vs Monte Carlo should agree within a few std errors
    opt = Option(S0=100, K=105, T=1.0, r=0.03, sigma=0.20, option_type=OptionType.CALL)

    bs_price = opt.price_bs()
    mc_price, mc_se = opt.price_mc(n_sims=200_000, seed=42)

    print(f"Black-Scholes price : {bs_price:.4f}")
    print(f"Monte Carlo price   : {mc_price:.4f}  (std error: {mc_se:.4f})")
    print("Greeks:")
    for name, value in opt.greeks().items():
        print(f"  {name:<6}: {value:.4f}")
