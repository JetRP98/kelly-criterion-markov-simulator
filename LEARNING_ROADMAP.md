# Learning Roadmap: Understanding This Project From Zero

You're a Python beginner. This roadmap tells you exactly what to learn,
in what order, so you can fully understand and explain every line of this code.

**Total time: ~3 weeks at 1 hour/day**
**Goal: understand the code deeply enough to explain it in a 20-minute interview deep-dive**

---

## Week 1: Python Fundamentals You Need

Work through these in order. Each one is a prerequisite for the next.

### Day 1–2: Variables, Types, Basic Math
**What to learn:**
- Variables: `x = 5`, `name = "Rohan"`, `rate = 0.55`
- Basic types: `int`, `float`, `str`, `bool`
- Math operators: `+`, `-`, `*`, `/`, `**` (power), `%` (modulo)

**In this project it appears as:**
```python
q = 1.0 - p          # q is a float; 1.0 - 0.55 = 0.45
f_star = (p * b - q) / b    # arithmetic using variables
```

**Resource:** Python.org tutorial Ch.1–3 (free) or W3Schools Python Intro

---

### Day 3: Functions
**What to learn:**
- `def function_name(arg1, arg2):` to define
- `return value` to give back a result
- Calling a function: `result = kelly_fraction(0.55, 1.0)`
- Default arguments: `def foo(x, n=100):`

**In this project it appears as:**
```python
def kelly_fraction(p: float, b: float) -> float:
    q = 1.0 - p
    f_star = (p * b - q) / b
    return max(0.0, f_star)
```
The `: float` and `-> float` are just type hints — they're optional labels
that tell you what type goes in and comes out.

---

### Day 4–5: Lists, Loops, Conditionals
**What to learn:**
- Lists: `prices = [100, 102, 99, 105]`
- `for` loops: `for x in prices:` and `for i in range(10):`
- `if/elif/else` conditionals
- `while` loops (less common but good to know)

**In this project it appears as:**
```python
for t in range(1, n_steps):      # loop from 1 to n_steps-1
    if use_regimes:               # conditional: choose which path
        step_mu = regime['mu']
    else:
        step_mu = mu
```

---

### Day 6–7: NumPy Arrays (THE most important library for this project)
**What to learn:**
- `import numpy as np`
- Creating arrays: `np.array([1, 2, 3])`, `np.zeros((5, 3))`, `np.ones(10)`
- Array arithmetic: `arr * 2`, `arr + 1`, `arr1 * arr2` (element-wise)
- Slicing: `arr[0]` (first element), `arr[:, 0]` (all rows, column 0)
- Key functions: `np.mean()`, `np.median()`, `np.std()`, `np.min()`, `np.max()`
- Random numbers: `np.random.normal(mu, sigma, n)` → n random draws

**This is 80% of the project. Spend 2 full days here.**

**In this project it appears as:**
```python
# Pre-allocate a 2D array: n_paths rows, n_steps columns, filled with 1.0
wealth_paths = np.ones((n_paths, n_steps))

# Generate random returns for ALL paths at once (no loop needed!)
returns = np.random.normal(step_mu, step_sigma, n_paths)

# Update wealth for all paths simultaneously
wealth_paths[:, t] = wealth_paths[:, t-1] * (1.0 + portfolio_returns)

# Compute median across paths (axis=0 means "across rows at each column")
median_path = np.median(wealth_paths, axis=0)
```

**Resource:** NumPy Quickstart Tutorial (numpy.org/doc) — do the whole thing

---

## Week 2: Understanding the Math

You don't need to derive these. You need to explain them intuitively.

### Kelly Criterion (Day 8–9)
**Read:** Wikipedia "Kelly Criterion" — read the whole page
**Then answer these out loud:**
1. "What does the Kelly fraction represent?" → the optimal fraction of capital to bet
2. "Why does overbetting lead to ruin?" → because compound losses dominate gains
3. "What's the difference between f=1.0 and f=2.0?" → 2x Kelly has negative expected log return
4. "Why do practitioners use half-Kelly?" → same growth, much lower variance and drawdown
5. "What does `f* = μ/σ²` mean intuitively?" → higher edge (μ) → bet more; higher risk (σ²) → bet less

### Markov Chains (Day 10–11)
**Read:** 3Blue1Brown "Markov Chains" on YouTube (15 min video)
**Then answer these out loud:**
1. "What is a Markov chain?" → system where next state depends only on current state
2. "What is the transition matrix?" → probability of going from each state to each other state
3. "What is the stationary distribution?" → long-run fraction of time in each state
4. "Why use Markov chains for market regimes?" → markets have persistent states (bull/bear)
5. "How did you estimate your transition probabilities?" → based on historical regime persistence

### Monte Carlo Simulation (Day 12–13)
**Concept:** simulate a random process thousands of times to estimate probabilities
**Then answer these out loud:**
1. "Why simulate 4,000 paths instead of 1?" → one path is noise; many paths show the distribution
2. "What is ruin probability?" → fraction of simulated paths that fell below 10% of starting wealth
3. "What is the 10th percentile line on your wealth plot?" → 10% of paths ended below this value
4. "Why do you use log scale on the y-axis?" → exponential growth looks linear on log scale

---

## Week 3: Running and Defending the Code

### Day 14–15: Install Python and Run the Code
```bash
# Install Python: download from python.org
# Then in your terminal:
pip install numpy matplotlib scipy
python kelly_simulator.py
```
Run it. Look at the plots. Make sure you understand what each chart shows.

### Day 16–17: Change Parameters and Observe
Try these experiments and explain what you see:
- Change BULL regime mu to 0.002 (more edge) — what happens to optimal Kelly fraction?
- Change transition_matrix[0][2] to 0.15 (crashes are more likely) — what happens to ruin prob?
- Change N_PATHS to 100 vs 10000 — how does the median path change?
- Set use_regimes=False — compare to the regime-switching version

### Day 18–19: Practice Explaining Out Loud
Stand in front of a mirror or record yourself answering:
- "Walk me through what this project does."
- "Why did you choose the Kelly Criterion?"
- "What is a Markov chain and why did you use it?"
- "What did you find? What surprised you?"
- "What are the limitations of this model?"
- "How would you extend this?"

### Day 20–21: GitHub and README
1. Create a GitHub account at github.com
2. Create a new repository: "kelly-criterion-simulator"
3. Upload: `kelly_simulator.py` and `README.md`
4. Add the generated PNG plots as images
5. Make sure the README renders properly on GitHub

---

## What to Say in an Interview

### Opening (30 seconds)
> "I built a Kelly Criterion simulator that models optimal bet sizing under market
> regime uncertainty. The core idea is that Kelly's formula tells you the mathematically
> optimal fraction of capital to risk, and I extended it with a Markov regime-switching
> model where the market moves between bull, bear, and crash states."

### Technical depth (if asked)
> "The continuous Kelly fraction is μ/σ², so in a bull regime with higher expected
> return and lower volatility, you size up. In a crash regime you size down or go flat.
> The Markov chain determines which regime you're in — each state has transition
> probabilities to every other state, and I compute the stationary distribution via
> eigendecomposition to show long-run time spent in each regime."

### Results (always quantify)
> "The key finding is the asymmetry of overbetting: full Kelly gives the highest
> median final wealth but 2-3% ruin probability. Double Kelly — despite having
> positive expected return — leads to 40-50% ruin probability over one year.
> Half Kelly gives about 75% of full Kelly's growth with near-zero ruin probability,
> which is why practitioners standardly use fractional Kelly."

### Limitations (shows intellectual honesty)
> "The main limitation is that in practice μ and σ aren't known — you estimate them
> from data, and estimation error makes the true optimal fraction uncertain. A natural
> extension would be adding parameter uncertainty to the model."

---

## Python Resources (All Free)

| Resource | What it covers | Time |
|---|---|---|
| python.org tutorial | Variables, loops, functions | 3 hours |
| numpy.org quickstart | NumPy arrays (critical) | 2 hours |
| 3Blue1Brown "Essence of Linear Algebra" | Why matrix operations work | 3 hours |
| matplotlib.org tutorials | How to make plots | 1 hour |
| "Heard on the Street" Ch. 1 | Kelly/probability problems | ongoing |

