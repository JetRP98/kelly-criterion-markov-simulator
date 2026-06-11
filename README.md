# Kelly Criterion Portfolio Simulator with Markov Regime Switching

A quantitative finance simulation exploring optimal bet sizing under market regime uncertainty.

## What This Project Does

The **Kelly Criterion** answers a fundamental question in trading: *given a known edge, what fraction of your capital should you risk on each trade to maximize long-run wealth?*

Bet too little — you leave money on the table. Bet too much — you blow up.

This project extends the classic Kelly model with a **Markov regime-switching framework**, where the market cycles between Bull, Bear, and Crash states with known transition probabilities. The optimal bet size changes with the regime — and this simulator shows exactly how.

---

## Key Concepts Implemented

### 1. Kelly Criterion (Binary & Continuous)
- **Binary Kelly**: `f* = (p*b - q) / b` — for discrete win/loss bets
- **Continuous Kelly**: `f* = μ / σ²` — for normally distributed returns (stocks)
- **Multi-asset Kelly**: `f* = Σ⁻¹ μ` — portfolio generalization via covariance matrix inversion

### 2. Markov Regime Switching
Three market states with empirically-motivated parameters:

| Regime | Daily μ | Daily σ | Intuition |
|--------|---------|---------|-----------|
| Bull   | +0.08%  | 0.80%   | Low vol, steady gains |
| Bear   | -0.02%  | 1.80%   | Elevated vol, slight drift down |
| Crash  | -0.30%  | 3.50%   | Crisis regime — fat tails |

Transition matrix (rows = from, cols = to):

```
         Bull   Bear   Crash
Bull  [ 0.97,  0.02,  0.01 ]
Bear  [ 0.05,  0.92,  0.03 ]
Crash [ 0.10,  0.40,  0.50 ]
```

**Stationary distribution** (long-run % of time in each regime): computed via eigendecomposition.

### 3. Monte Carlo Simulation Engine
- Simulates 4,000+ independent portfolio paths over 252 trading days (1 year)
- Each path samples returns from the current regime's distribution
- Tracks: wealth, drawdown, ruin events

### 4. Analysis & Visualization
- **Wealth path plots**: 200 individual paths + median + 10th/90th percentile bands
- **Kelly curve**: median final wealth, ruin probability, and max drawdown vs. fraction used
- **Regime path**: Markov chain visualization with stationary distribution

---

## Results & Key Findings

| Strategy | Median Final Wealth | Ruin Probability | Median Max Drawdown |
|---|---|---|---|
| Quarter Kelly (0.25x) | ~1.05x | ~0% | ~8% |
| Half Kelly (0.5x) | ~1.12x | ~0% | ~15% |
| Full Kelly (1.0x) | ~1.18x | ~2% | ~35% |
| Double Kelly (2.0x) | ~0.90x | ~45% | ~75% |

**Key insight**: overbetting (2x+ Kelly) leads to near-certain ruin even with a positive expected return. Half-Kelly gives ~75% of full Kelly's growth with dramatically lower drawdowns — the standard choice in practice.

---

## How to Run

```bash
# Install dependencies
pip install numpy matplotlib scipy

# Run the simulation
python kelly_simulator.py
```

Outputs three PNG files:
- `kelly_wealth_paths.png` — 2×2 grid of wealth paths per strategy
- `kelly_curves.png` — Kelly curve analysis (growth/ruin/drawdown)
- `kelly_regimes.png` — Markov regime path visualization

---

## Mathematical Background

### Why Kelly Maximizes Log-Utility

Kelly's formula is derived by maximizing the **expected log of wealth**, not expected wealth directly. This is crucial:
- Maximizing E[W] ignores the risk of ruin
- Maximizing E[log W] automatically penalizes paths that go to zero
- The log-optimal strategy maximizes long-run geometric growth rate

### The Ruin Problem

For a strategy with fraction `f` and return distribution `r ~ N(μ, σ²)`:

```
E[log(1 + f*r)] ≈ f*μ - (f²*σ²)/2
```

This is maximized at `f* = μ/σ²`. Beyond `f = 2*f*`, the expected log return becomes negative — meaning long-run wealth *shrinks* with probability 1, even though expected wealth is positive.

### Stationary Distribution via Eigendecomposition

The long-run fraction of time in each regime is the stationary distribution `π` satisfying:
```
π = π * T
```
Computed as the eigenvector of `T^T` corresponding to eigenvalue 1, normalized to sum to 1.

---

## Skills Demonstrated

- **Python**: NumPy array operations, vectorized simulation, matplotlib visualization
- **Quantitative Finance**: Kelly criterion, position sizing, drawdown analysis
- **Stochastic Processes**: Markov chains, Monte Carlo simulation, regime switching
- **Statistics**: Eigendecomposition, percentile analysis, probability distributions

---

## Potential Extensions

- [ ] Add parameter uncertainty — estimate Kelly fraction from noisy data
- [ ] Implement fractional Kelly with drawdown constraints (practical risk management)
- [ ] Transaction costs and slippage impact on optimal sizing
- [ ] Compare to Sharpe-optimal sizing (different objective, different result)
- [ ] Calibrate transition matrix to real VIX regime data

---

*Built as part of a quant trading internship preparation project.*
