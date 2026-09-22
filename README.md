# Python Options Pricer

Object-oriented vanilla options pricer implemented in Python.

## Status

Core pricing engine implemented and validated:
- Black-Scholes closed-form pricing (call & put)
- Monte Carlo pricing under geometric Brownian motion, with standard error estimate
- Greeks: delta, gamma, vega, theta, rho

Black-Scholes and Monte Carlo prices agree within Monte Carlo standard error on a
basic test case (see `if __name__ == "__main__"` block in `options_pricer.py`).

**Next steps:** American option pricing (binomial tree / Longstaff-Schwartz),
implied volatility solver, unit tests, and a small CLI/notebook demo.

## Usage

```python
from options_pricer import Option, OptionType

opt = Option(S0=100, K=105, T=1.0, r=0.03, sigma=0.20, option_type=OptionType.CALL)

print(opt.price_bs())        # Black-Scholes price
print(opt.price_mc())        # (Monte Carlo price, standard error)
print(opt.greeks())          # dict of delta, gamma, vega, theta, rho
```

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## Run the demo

```bash
python options_pricer.py
```
