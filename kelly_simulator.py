import numpy as np                          # numerical computing library
import matplotlib.pyplot as plt             # plotting library
import matplotlib.gridspec as gridspec      # for complex subplot layouts
from scipy.stats import norm                # normal distribution functions
from typing import Tuple, List              # type hints (makes code readable)

# Set random seed so results are reproducible (same "random" numbers every run)
np.random.seed(42)

# Set a clean visual style for all plots
plt.style.use('seaborn-v0_8-whitegrid')


# SECTION 1: KELLY CRITERION MATHEMATICS

def kelly_fraction(p: float, b: float) -> float:
    """
    Compute the Kelly optimal bet fraction for a single binary outcome.

    THE FORMULA:  f* = (p*b - q) / b
    WHERE:
        p = probability of winning (e.g. 0.55 = 55% win rate)
        b = net odds (e.g. b=1.0 means you win $1 for every $1 bet)
        q = probability of losing = 1 - p

    INTUITION:
        If you have NO edge (p=0.5, b=1.0), Kelly says bet 0% — correct,
        because in a fair coin flip you can't grow wealth long-term.
        The more edge you have, the more Kelly tells you to bet.

    Args:
        p: win probability (float between 0 and 1)
        b: net odds received on a win

    Returns:
        f_star: optimal fraction of capital to bet (float)
    """
    q = 1.0 - p                             # loss probability
    f_star = (p * b - q) / b               # Kelly formula
    return max(0.0, f_star)                 # can't bet negative fraction


def kelly_fraction_continuous(mu: float, sigma: float) -> float:
    """
    Kelly fraction for CONTINUOUS returns (stock-like assets).

    When returns are normally distributed (as in a GBM stock model),
    the Kelly fraction simplifies to:

        f* = mu / sigma^2

    WHERE:
        mu    = expected return per period (e.g. 0.001 = 0.1% per day)
        sigma = standard deviation of returns (volatility)

    This is the version used for actual financial assets.

    Args:
        mu:    expected return per period
        sigma: standard deviation of returns per period

    Returns:
        f_star: Kelly-optimal leverage/fraction
    """
    if sigma == 0:
        return 0.0                          # avoid division by zero
    return mu / (sigma ** 2)               # Kelly formula for continuous returns


def multi_asset_kelly(mu_vec: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Multi-asset Kelly: find optimal weights for a PORTFOLIO of assets.

    For multiple assets, Kelly says to solve:
        f* = Sigma^(-1) * mu
    WHERE:
        Sigma^(-1) = inverse of the covariance matrix
        mu         = vector of expected returns

    This is closely related to mean-variance optimization (Markowitz)
    but Kelly maximizes geometric growth, not Sharpe ratio.

    Args:
        mu_vec:     1D array of expected returns for each asset
        cov_matrix: 2D covariance matrix of asset returns

    Returns:
        weights: 1D array of optimal Kelly fractions per asset
    """
    cov_inv = np.linalg.inv(cov_matrix)    # matrix inverse (numpy handles this)
    weights = cov_inv @ mu_vec              # matrix-vector multiplication
    return weights



# SECTION 2: MARKOV REGIME SWITCHING MODEL
# WHAT IS A MARKOV CHAIN?
#   A system where the future state depends ONLY on the current state,
#   not on the history. Each day, the market can be in:
#       State 0: BULL  (high returns, low volatility)
#       State 1: BEAR  (low/negative returns, high volatility)
#       State 2: CRASH (very negative returns, very high volatility)
#
# TRANSITION MATRIX:
#   Each row = current state, each column = next state
#   Entry [i,j] = probability of moving from state i to state j
#
#   Example: transition_matrix[0][1] = 0.05 means
#   "if today is a BULL day, there's a 5% chance tomorrow is BEAR"
#
# WHY THIS MATTERS FOR KELLY:
#   The optimal bet fraction changes with the regime.
#   In a bull market you should bet more; in a crash, less or zero.

# Market regime parameters
REGIMES = {
    'BULL':  {'mu': 0.0008,  'sigma': 0.008,  'color': '#2ecc71', 'label': 'Bull'},
    'BEAR':  {'mu': -0.0002, 'sigma': 0.018,  'color': '#e67e22', 'label': 'Bear'},
    'CRASH': {'mu': -0.003,  'sigma': 0.035,  'color': '#e74c3c', 'label': 'Crash'},
}

# Transition matrix: rows=from, cols=to, order=[BULL, BEAR, CRASH]
# Each row must sum to 1.0 (probabilities)
TRANSITION_MATRIX = np.array([
    [0.97, 0.02, 0.01],   # From BULL:  97% stay bull, 2% → bear, 1% → crash
    [0.05, 0.92, 0.03],   # From BEAR:  5% → bull, 92% stay bear, 3% → crash
    [0.10, 0.40, 0.50],   # From CRASH: 10% → bull, 40% → bear, 50% stay crash
])


def simulate_regime_path(n_steps: int, initial_regime: int = 0) -> np.ndarray:
    """
    Simulate a sequence of market regimes using the Markov transition matrix.

    HOW IT WORKS:
        Start in regime `initial_regime`.
        Each step, draw a random number and use the transition probabilities
        to decide which regime comes next.
        This is the core Markov chain simulation loop.

    Args:
        n_steps:        number of time steps to simulate
        initial_regime: starting regime (0=Bull, 1=Bear, 2=Crash)

    Returns:
        regimes: array of regime indices for each time step
    """
    regimes = np.zeros(n_steps, dtype=int)  # pre-allocate array (faster than appending)
    regimes[0] = initial_regime              # set starting state

    regime_params = list(REGIMES.values())   # convert dict to list for indexing

    for t in range(1, n_steps):
        current = regimes[t - 1]             # what regime are we in now?

        # np.random.choice: pick from [0,1,2] with given probabilities
        # TRANSITION_MATRIX[current] gives the probability of going to each next state
        regimes[t] = np.random.choice(
            a=3,                                    # pick from 0, 1, or 2
            p=TRANSITION_MATRIX[current]            # with these probabilities
        )

    return regimes


def get_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Find the long-run (stationary) distribution of the Markov chain.

    The stationary distribution π satisfies: π = π * T
    This tells us: in the long run, what fraction of time is spent in each regime?

    MATH: We find the left eigenvector corresponding to eigenvalue 1.
    The stationary distribution is the eigenvector normalized to sum to 1.

    Args:
        transition_matrix: square Markov transition matrix

    Returns:
        stationary: array of long-run probabilities for each state
    """
    # Compute eigenvalues and eigenvectors of the TRANSPOSE
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

    # Find the eigenvector where eigenvalue ≈ 1.0 (the stationary one)
    # np.argmin finds the index of the smallest value
    idx = np.argmin(np.abs(eigenvalues - 1.0))

    # Extract that eigenvector and take the real part (might have tiny imaginary parts)
    stationary = np.real(eigenvectors[:, idx])

    # Normalize so it sums to 1 (it's a probability distribution)
    stationary = stationary / stationary.sum()

    return stationary


# SECTION 3: PORTFOLIO SIMULATION ENGINE

def simulate_portfolio(
    kelly_fraction_used: float,
    n_steps: int = 252,
    n_paths: int = 5000,
    use_regimes: bool = True,
    mu: float = 0.0005,
    sigma: float = 0.012,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate portfolio wealth paths using a given Kelly fraction.

    KEY CONCEPTS:
        - We simulate many (n_paths) independent portfolio journeys
        - Each step, the portfolio return = kelly_fraction * asset_return
        - If we use regimes, the asset return distribution changes each step
        - We track the portfolio value over time for each path

    HOW PORTFOLIO UPDATE WORKS:
        wealth[t] = wealth[t-1] * (1 + f * r_t)
        WHERE:
            f   = kelly fraction we're using (the "bet size")
            r_t = random return of the asset at time t

        If f=0.5 and the asset goes up 2%, portfolio goes up 1%.
        If f=2.0 (leveraged) and asset goes up 2%, portfolio goes up 4%.
        But if f=2.0 and asset goes DOWN 60%, portfolio is WIPED OUT.

    Args:
        kelly_fraction_used: fraction of Kelly to use (1.0=full Kelly, 0.5=half Kelly)
        n_steps:             number of trading days to simulate
        n_paths:             number of independent simulation paths
        use_regimes:         if True, use Markov regime switching
        mu:                  base expected daily return (used if use_regimes=False)
        sigma:               base daily volatility (used if use_regimes=False)

    Returns:
        wealth_paths: 2D array of shape (n_paths, n_steps) — wealth over time
        regime_path:  1D array of regimes (only meaningful if use_regimes=True)
    """
    regime_params = list(REGIMES.values())

    # Pre-allocate: create a 2D array of zeros — n_paths rows, n_steps columns
    # Each row is one simulation path; each column is one time step
    wealth_paths = np.ones((n_paths, n_steps))  # start with wealth = 1.0

    # Simulate one regime path (shared across all portfolio paths for realism)
    regime_path = simulate_regime_path(n_steps)

    for t in range(1, n_steps):             # loop through each time step

        if use_regimes:
            # Get the return parameters for today's regime
            regime = regime_params[regime_path[t]]
            step_mu = regime['mu']
            step_sigma = regime['sigma']
        else:
            step_mu = mu
            step_sigma = sigma

        # Compute the Kelly-optimal fraction for THIS regime's parameters
        f_optimal = kelly_fraction_continuous(step_mu, step_sigma)

        # Scale by the fraction of Kelly we've chosen to use
        f_actual = kelly_fraction_used * f_optimal

        # Clamp leverage to avoid extreme values (practical risk management)
        f_actual = np.clip(f_actual, -2.0, 5.0)

        # Generate random returns for all n_paths simultaneously
        # np.random.normal(mu, sigma, n_paths) → draws n_paths random values
        # from a normal distribution with mean=mu, std=sigma
        returns = np.random.normal(step_mu, step_sigma, n_paths)

        # Portfolio return = leverage * asset return
        portfolio_returns = f_actual * returns

        # Update wealth: new_wealth = old_wealth * (1 + portfolio_return)
        # wealth_paths[:, t-1] = previous time step for ALL paths (the colon means "all rows")
        wealth_paths[:, t] = wealth_paths[:, t - 1] * (1.0 + portfolio_returns)

        # Floor wealth at near-zero (can't go below 0 in practice — margin call)
        wealth_paths[:, t] = np.maximum(wealth_paths[:, t], 1e-6)

    return wealth_paths, regime_path


def compute_ruin_probability(wealth_paths: np.ndarray, ruin_threshold: float = 0.1) -> float:
    """
    Compute the probability of 'ruin' — portfolio falling below a threshold.

    RUIN is defined as: wealth dropping below `ruin_threshold` at ANY point.
    (e.g. losing 90% of starting capital = ruin if threshold=0.1)

    Args:
        wealth_paths:    2D array (n_paths x n_steps) of simulated wealth
        ruin_threshold:  fraction of starting capital below which = ruin

    Returns:
        ruin_prob: fraction of paths that experienced ruin (float 0 to 1)
    """
    # For each path, find the minimum wealth ever reached
    # np.min(axis=1) → take the minimum across columns (time), for each row (path)
    min_wealth_per_path = np.min(wealth_paths, axis=1)

    # Count how many paths went below the ruin threshold
    # (min_wealth_per_path < ruin_threshold) creates a boolean array (True/False)
    # np.mean of a boolean array = fraction of True values
    ruin_prob = np.mean(min_wealth_per_path < ruin_threshold)

    return ruin_prob


def compute_drawdown(wealth_path: np.ndarray) -> np.ndarray:
    """
    Compute the drawdown series for a single wealth path.

    DRAWDOWN = how far below the previous peak are we?
    Formula: DD[t] = (peak_so_far[t] - wealth[t]) / peak_so_far[t]

    A drawdown of 0.3 means we're 30% below our highest point.
    Max drawdown = the worst drawdown experienced over the entire path.

    Args:
        wealth_path: 1D array of portfolio values over time

    Returns:
        drawdown: 1D array of drawdown values (0 = at peak, 1 = total loss)
    """
    # np.maximum.accumulate → running maximum (peak so far at each time step)
    running_peak = np.maximum.accumulate(wealth_path)

    # Drawdown = how far below peak
    drawdown = (running_peak - wealth_path) / running_peak

    return drawdown


# SECTION 4: ANALYSIS FUNCTIONS

def run_kelly_fraction_sweep(
    fractions: List[float],
    n_steps: int = 252,
    n_paths: int = 3000,
) -> dict:
    """
    Sweep across different Kelly fractions and record key statistics.

    This produces the data for the main Kelly curve plot:
    for each fraction f, what is the median final wealth, ruin probability,
    and median max drawdown?

    Args:
        fractions: list of Kelly fractions to test (e.g. [0.25, 0.5, 1.0, 2.0])
        n_steps:   trading days per simulation
        n_paths:   number of paths per fraction

    Returns:
        results: dict with arrays for each metric across fractions
    """
    results = {
        'fractions': fractions,
        'median_final_wealth': [],
        'mean_final_wealth': [],
        'ruin_prob': [],
        'median_max_drawdown': [],
        'pct_10_wealth': [],          # 10th percentile (bad outcome)
        'pct_90_wealth': [],          # 90th percentile (good outcome)
    }

    print("Running Kelly fraction sweep...")
    for i, f in enumerate(fractions):
        print(f"  Testing f = {f:.2f}x Kelly ({i+1}/{len(fractions)})")

        # Simulate portfolio paths for this Kelly fraction
        wealth_paths, _ = simulate_portfolio(
            kelly_fraction_used=f,
            n_steps=n_steps,
            n_paths=n_paths,
            use_regimes=True,
        )

        # Final wealth = last column of wealth_paths
        final_wealth = wealth_paths[:, -1]

        # Max drawdown for each path
        max_drawdowns = np.array([
            np.max(compute_drawdown(wealth_paths[i, :]))
            for i in range(n_paths)
        ])

        # Record statistics
        # np.median/mean/percentile compute these stats across the n_paths dimension
        results['median_final_wealth'].append(np.median(final_wealth))
        results['mean_final_wealth'].append(np.mean(final_wealth))
        results['ruin_prob'].append(compute_ruin_probability(wealth_paths))
        results['median_max_drawdown'].append(np.median(max_drawdowns))
        results['pct_10_wealth'].append(np.percentile(final_wealth, 10))
        results['pct_90_wealth'].append(np.percentile(final_wealth, 90))

    # Convert lists to numpy arrays for easier plotting
    for key in results:
        if key != 'fractions':
            results[key] = np.array(results[key])

    return results


# SECTION 5: VISUALIZATION

def plot_wealth_paths(ax, wealth_paths: np.ndarray, regime_path: np.ndarray,
                      title: str, fraction_label: str, color: str):
    """
    Plot a sample of simulated wealth paths with regime background shading.

    Args:
        ax:            matplotlib axes object to plot on
        wealth_paths:  2D array of wealth over time
        regime_path:   1D array of regime indices
        title:         plot title
        fraction_label: label for the legend
        color:         line color
    """
    n_steps = wealth_paths.shape[1]
    t = np.arange(n_steps)                 # time axis: [0, 1, 2, ..., n_steps-1]

    # Shade background by regime
    regime_colors = ['#d5f5e3', '#fdebd0', '#fadbd8']   # light green/orange/red
    regime_params = list(REGIMES.values())

    prev_regime = regime_path[0]
    start = 0
    for i in range(1, n_steps):
        if regime_path[i] != prev_regime or i == n_steps - 1:
            # ax.axvspan: shade a vertical strip from x=start to x=i
            ax.axvspan(start, i, alpha=0.15,
                       color=regime_colors[prev_regime], linewidth=0)
            start = i
            prev_regime = regime_path[i]

    # Plot a random sample of 200 paths (plotting all 5000 would be too slow)
    sample_indices = np.random.choice(wealth_paths.shape[0], size=200, replace=False)
    for idx in sample_indices:
        ax.plot(t, wealth_paths[idx], alpha=0.08, linewidth=0.5, color=color)

    # Plot median path (thicker line) — np.median(axis=0) = median across paths at each time
    median_path = np.median(wealth_paths, axis=0)
    ax.plot(t, median_path, linewidth=2.5, color=color, label=f'Median ({fraction_label})', zorder=5)

    # Plot 10th and 90th percentile bands
    p10 = np.percentile(wealth_paths, 10, axis=0)
    p90 = np.percentile(wealth_paths, 90, axis=0)
    ax.fill_between(t, p10, p90, alpha=0.15, color=color, label='10th–90th pct')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Trading Days', fontsize=10)
    ax.set_ylabel('Portfolio Value (Starting = 1.0)', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_yscale('log')                    # log scale makes exponential growth linear
    ax.set_xlim(0, n_steps)


def plot_kelly_curve(ax_growth, ax_ruin, ax_drawdown, results: dict):
    """
    Plot the classic Kelly curves: growth, ruin probability, and drawdown vs fraction.

    The KEY insight these plots show:
        - Growth peaks at f=1.0 (full Kelly) and falls off sharply above
        - Ruin probability explodes above f=2.0 (overbetting)
        - Drawdown increases monotonically with f (more leverage = worse drawdowns)

    Args:
        ax_growth:   axes for median final wealth
        ax_ruin:     axes for ruin probability
        ax_drawdown: axes for median max drawdown
        results:     dict from run_kelly_fraction_sweep()
    """
    fractions = results['fractions']

    # --- Growth curve ---
    ax_growth.plot(fractions, results['median_final_wealth'],
                   color='#2ecc71', linewidth=2.5, marker='o', markersize=5, label='Median')
    ax_growth.fill_between(fractions,
                            results['pct_10_wealth'],
                            results['pct_90_wealth'],
                            alpha=0.2, color='#2ecc71', label='10th–90th pct')
    ax_growth.axvline(x=1.0, color='#2c3e50', linestyle='--', linewidth=1.5, label='Full Kelly (f=1)')
    ax_growth.axvline(x=0.5, color='#7f8c8d', linestyle=':', linewidth=1.5, label='Half Kelly (f=0.5)')
    ax_growth.set_title('Portfolio Growth vs. Kelly Fraction', fontsize=11, fontweight='bold')
    ax_growth.set_xlabel('Fraction of Kelly Used', fontsize=10)
    ax_growth.set_ylabel('Median Final Wealth (log scale)', fontsize=9)
    ax_growth.set_yscale('log')
    ax_growth.legend(fontsize=8)

    # --- Ruin probability curve ---
    ax_ruin.plot(fractions, results['ruin_prob'] * 100,
                 color='#e74c3c', linewidth=2.5, marker='s', markersize=5)
    ax_ruin.axvline(x=1.0, color='#2c3e50', linestyle='--', linewidth=1.5)
    ax_ruin.axvline(x=0.5, color='#7f8c8d', linestyle=':', linewidth=1.5)
    ax_ruin.axhline(y=5.0, color='#e74c3c', linestyle='-.', linewidth=1, alpha=0.5, label='5% ruin threshold')
    ax_ruin.set_title('Ruin Probability vs. Kelly Fraction', fontsize=11, fontweight='bold')
    ax_ruin.set_xlabel('Fraction of Kelly Used', fontsize=10)
    ax_ruin.set_ylabel('Ruin Probability (%)', fontsize=10)
    ax_ruin.legend(fontsize=8)
    ax_ruin.set_ylim(0, 100)

    # --- Drawdown curve ---
    ax_drawdown.plot(fractions, results['median_max_drawdown'] * 100,
                     color='#e67e22', linewidth=2.5, marker='^', markersize=5)
    ax_drawdown.axvline(x=1.0, color='#2c3e50', linestyle='--', linewidth=1.5, label='Full Kelly')
    ax_drawdown.axvline(x=0.5, color='#7f8c8d', linestyle=':', linewidth=1.5, label='Half Kelly')
    ax_drawdown.set_title('Median Max Drawdown vs. Kelly Fraction', fontsize=11, fontweight='bold')
    ax_drawdown.set_xlabel('Fraction of Kelly Used', fontsize=10)
    ax_drawdown.set_ylabel('Median Max Drawdown (%)', fontsize=10)
    ax_drawdown.legend(fontsize=8)
    ax_drawdown.set_ylim(0, 100)


def plot_regime_analysis(ax, n_steps: int = 1000):
    """
    Visualize the Markov regime switching: show simulated regime path
    and the stationary distribution.

    Args:
        ax:      matplotlib axes
        n_steps: length of regime path to simulate
    """
    regime_path = simulate_regime_path(n_steps)
    stationary = get_stationary_distribution(TRANSITION_MATRIX)

    regime_params = list(REGIMES.values())
    regime_colors = [r['color'] for r in regime_params]
    regime_labels = [r['label'] for r in regime_params]

    t = np.arange(n_steps)

    # Plot regime as a step function
    ax.step(t, regime_path, where='post', color='#2c3e50', linewidth=0.8, alpha=0.7)

    # Shade background by regime
    for i, (color, label) in enumerate(zip(regime_colors, regime_labels)):
        mask = regime_path == i
        # Fill regions where this regime is active
        ax.fill_between(t, i - 0.4, i + 0.4, where=mask,
                        color=color, alpha=0.6, label=f'{label} ({stationary[i]*100:.1f}% of time)')

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(regime_labels)
    ax.set_xlabel('Time Steps', fontsize=10)
    ax.set_title('Markov Regime Switching Path\n(Stationary distribution shown in legend)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, n_steps)


# SECTION 6: MAIN — RUN EVERYTHING

def main():
    """
    Main function: runs the full analysis and generates all plots.

    WHAT WE PRODUCE:
        Figure 1: Wealth path comparison (underbetting vs half-Kelly vs full Kelly vs overbetting)
        Figure 2: Kelly curves (growth, ruin, drawdown vs fraction)
        Figure 3: Regime analysis (Markov chain visualization)
    """

    print("=" * 70)
    print("  KELLY CRITERION & MARKOV REGIME SWITCHING SIMULATOR")
    print("=" * 70)

    # --- Print key statistics ---
    print("\n[1] Computing Kelly fractions by regime:")
    regime_params = list(REGIMES.values())
    regime_names = list(REGIMES.keys())
    for name, params in REGIMES.items():
        f_kelly = kelly_fraction_continuous(params['mu'], params['sigma'])
        print(f"    {name:6s}: mu={params['mu']:.4f}, sigma={params['sigma']:.4f} → Kelly f* = {f_kelly:.3f}x")

    print("\n[2] Stationary distribution of regimes (long-run % of time in each):")
    stationary = get_stationary_distribution(TRANSITION_MATRIX)
    for i, (name, pct) in enumerate(zip(REGIMES.keys(), stationary)):
        print(f"    {name:6s}: {pct*100:.1f}%")

    # --- Simulate wealth paths for four strategies ---
    print("\n[3] Simulating wealth paths...")
    N_STEPS = 252                           # one trading year
    N_PATHS = 4000                          # number of simulation paths

    configs = [
        (0.25, 'Quarter Kelly', '#3498db'),
        (0.50, 'Half Kelly',    '#2ecc71'),
        (1.00, 'Full Kelly',    '#f39c12'),
        (2.00, 'Double Kelly',  '#e74c3c'),
    ]

    sim_results = {}
    for fraction, label, color in configs:
        print(f"    Simulating {label} (f={fraction})...")
        wealth_paths, regime_path = simulate_portfolio(
            kelly_fraction_used=fraction,
            n_steps=N_STEPS,
            n_paths=N_PATHS,
            use_regimes=True,
        )
        final_wealth = wealth_paths[:, -1]
        ruin_prob = compute_ruin_probability(wealth_paths)
        max_dd = np.median([np.max(compute_drawdown(wealth_paths[i, :])) for i in range(N_PATHS)])

        sim_results[label] = {
            'wealth_paths': wealth_paths,
            'regime_path': regime_path,
            'color': color,
            'fraction': fraction,
            'median_final': np.median(final_wealth),
            'ruin_prob': ruin_prob,
            'max_drawdown': max_dd,
        }
        print(f"      Median final wealth: {np.median(final_wealth):.3f}x | "
              f"Ruin prob: {ruin_prob*100:.1f}% | Max DD: {max_dd*100:.1f}%")

    # --- Kelly fraction sweep ---
    print("\n[4] Running Kelly fraction sweep (this takes ~30 seconds)...")
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    sweep_results = run_kelly_fraction_sweep(fractions, n_steps=N_STEPS, n_paths=2000)


    # FIGURE 1: Wealth paths

    print("\n[5] Generating plots...")
    fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig1.suptitle(
        'Kelly Criterion Portfolio Paths with Markov Regime Switching\n'
        'Green=Bull  |  Orange=Bear  |  Red=Crash',
        fontsize=14, fontweight='bold', y=1.01
    )

    for ax, (label, data) in zip(axes.flatten(), sim_results.items()):
        plot_wealth_paths(
            ax=ax,
            wealth_paths=data['wealth_paths'],
            regime_path=data['regime_path'],
            title=f"{label} (f = {data['fraction']}x Kelly)\n"
                  f"Median: {data['median_final']:.2f}x  |  "
                  f"Ruin: {data['ruin_prob']*100:.1f}%  |  "
                  f"Max DD: {data['max_drawdown']*100:.1f}%",
            fraction_label=label,
            color=data['color'],
        )

    plt.tight_layout()
    fig1.savefig('kelly_wealth_paths.png', dpi=150, bbox_inches='tight')
    print("    Saved: kelly_wealth_paths.png")


    # FIGURE 2: Kelly curves
   
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(
        'Kelly Criterion Analysis: Growth, Ruin, and Drawdown vs. Bet Fraction',
        fontsize=13, fontweight='bold'
    )
    plot_kelly_curve(axes2[0], axes2[1], axes2[2], sweep_results)
    plt.tight_layout()
    fig2.savefig('kelly_curves.png', dpi=150, bbox_inches='tight')
    print("    Saved: kelly_curves.png")

   
    # FIGURE 3: Regime analysis
   
    fig3, ax3 = plt.subplots(figsize=(16, 3))
    fig3.suptitle('Markov Chain Regime Path Simulation', fontsize=13, fontweight='bold')
    plot_regime_analysis(ax3, n_steps=500)
    plt.tight_layout()
    fig3.savefig('kelly_regimes.png', dpi=150, bbox_inches='tight')
    print("    Saved: kelly_regimes.png")

    print("\n" + "=" * 70)
    print("  DONE. Check the PNG files for your plots.")
    print("  Key takeaways to explain in interviews:")
    print("  1. Full Kelly maximizes long-run growth but has high variance")
    print("  2. Half Kelly gives ~75% of the growth with far lower drawdowns")
    print("  3. Overbetting (2x+ Kelly) leads to near-certain ruin over time")
    print("  4. Regime switching shows why bet sizing must adapt to market state")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
