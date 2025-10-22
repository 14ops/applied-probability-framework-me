# Theoretical Background

Mathematical foundations and probabilistic theory underlying the Applied Probability Framework.

## Table of Contents

1. [Monte Carlo Methods](#monte-carlo-methods)
2. [Bayesian Inference](#bayesian-inference)
3. [Kelly Criterion](#kelly-criterion)
4. [Convergence Theory](#convergence-theory)
5. [Variance Reduction](#variance-reduction)
6. [Multi-Armed Bandits](#multi-armed-bandits)

---

## Monte Carlo Methods

### Foundations

Monte Carlo methods estimate quantities of interest through repeated random sampling. For an expectation:

```
E[X] = ∫ x f(x) dx ≈ (1/n) Σᵢ xᵢ
```

where x₁, x₂, ..., xₙ are independent samples from f(x).

### Law of Large Numbers

By the Strong Law of Large Numbers:

```
(1/n) Σᵢ Xᵢ →ᵃ·ˢ· E[X]  as n → ∞
```

The sample mean converges almost surely to the true expectation.

### Central Limit Theorem

The sampling distribution of the mean is approximately normal:

```
√n (X̄ₙ - μ) → N(0, σ²)
```

This enables confidence interval construction:

```
CI₉₅% = X̄ₙ ± 1.96 × (σ/√n)
```

**Implementation:**

```python
def calculate_confidence_interval(data, confidence=0.95):
    """Calculate CI using t-distribution for finite samples."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h
```

### Convergence Rate

Monte Carlo error decreases as O(1/√n):

```
E[|X̄ₙ - μ|] = O(1/√n)
```

To halve the error, 4× the samples are needed. This motivates:
1. **Variance reduction techniques**
2. **Adaptive sampling** (stop when converged)
3. **Stratified sampling**

---

## Bayesian Inference

### Beta-Binomial Model

For binary outcomes (win/loss), the Beta distribution is the **conjugate prior** for the Binomial likelihood.

**Prior:**
```
p ~ Beta(α, β)
f(p|α,β) = [Γ(α+β)/(Γ(α)Γ(β))] p^(α-1) (1-p)^(β-1)
```

**Likelihood:**
```
X|p ~ Binomial(n, p)
P(X=k|p,n) = C(n,k) p^k (1-p)^(n-k)
```

**Posterior:**
```
p|X ~ Beta(α+k, β+n-k)
```

where k successes observed in n trials.

### Posterior Statistics

**Mean (Point Estimate):**
```
E[p|X] = (α+k)/(α+β+n)
```

**Mode (MAP Estimate):**
```
mode(p|X) = (α+k-1)/(α+β+n-2)    if α,β > 1
```

**Variance:**
```
Var[p|X] = (α+k)(β+n-k) / [(α+β+n)²(α+β+n+1)]
```

**Credible Interval:**

For 95% credible interval:
```
CI₉₅% = [Beta⁻¹(0.025|α+k, β+n-k), Beta⁻¹(0.975|α+k, β+n-k)]
```

### Prior Selection

**Uninformative Prior:**
```
α = β = 1  (Uniform[0,1])
```

**Jeffrey's Prior:**
```
α = β = 0.5  (Invariant to reparameterization)
```

**Informative Prior:**
```
α = μ × ν,  β = (1-μ) × ν
```
where μ is prior belief and ν is prior "sample size"

**Implementation:**

```python
class BetaEstimator:
    """Bayesian estimator for Bernoulli processes."""
    
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
    
    def update(self, successes, failures):
        """Bayesian update."""
        self.alpha += successes
        self.beta += failures
    
    def estimate(self):
        """Posterior mean."""
        return self.alpha / (self.alpha + self.beta)
    
    def variance(self):
        """Posterior variance."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))
```

### Bayesian vs. Frequentist

| Aspect | Bayesian | Frequentist |
|--------|----------|-------------|
| Parameters | Random variables | Fixed unknowns |
| Inference | Posterior distribution | Point estimate + CI |
| Uncertainty | Credible intervals | Confidence intervals |
| Prior knowledge | Incorporated via prior | Not used |
| Small samples | Regularized by prior | Can be unstable |

---

## Kelly Criterion

### Derivation

Given a bet with:
- Probability p of winning
- Odds b:1 (win $b for every $1 bet)
- Probability q = 1-p of losing

The **Kelly Criterion** maximizes expected log wealth:

```
max_f E[log(W_{t+1})]
```

where W_{t+1} = W_t(1 + fb) with prob p, W_t(1 - f) with prob q.

**Solution:**
```
f* = (bp - q) / b = (p(b+1) - 1) / b
```

For even money (b=1):
```
f* = 2p - 1
```

### Fractional Kelly

Full Kelly can be aggressive. **Fractional Kelly** reduces risk:

```
f_actual = f_Kelly × fraction
```

Common fractions:
- **Half Kelly** (0.5): Reduces volatility by 50%, growth by 25%
- **Quarter Kelly** (0.25): Conservative, stable growth
- **Full Kelly** (1.0): Maximum growth but high volatility

**Implementation:**

```python
def kelly_criterion(p, b=1.0, fraction=0.25):
    """
    Calculate fractional Kelly bet size.
    
    Args:
        p: Probability of winning
        b: Odds (net odds, not including stake)
        fraction: Fraction of Kelly to bet
    
    Returns:
        Fraction of bankroll to bet
    """
    kelly_full = (p * (b + 1) - 1) / b
    return max(0, kelly_full * fraction)
```

### Kelly with Uncertainty

When p is uncertain (estimated from data), using p̂ in place of true p leads to **overbet risk**.

**Solutions:**

1. **Bayesian Kelly**: Use posterior distribution
```
f_Bayes = E_posterior[f(p)] = ∫ f(p) π(p|data) dp
```

2. **Conservative Kelly**: Use lower confidence bound
```
f_conservative = f(p_lower_ci)
```

3. **Adaptive Kelly**: Adjust fraction based on uncertainty
```
fraction = base_fraction × (1 - CI_width/threshold)
```

**Implementation:**

```python
class AdaptiveKelly:
    def __init__(self, base_fraction=0.25):
        self.base_fraction = base_fraction
        self.estimator = BetaEstimator()
    
    def get_bet_size(self, b=1.0):
        p = self.estimator.estimate()
        ci_lower, ci_upper = self.estimator.confidence_interval()
        ci_width = ci_upper - ci_lower
        
        # Reduce fraction when uncertain
        if ci_width > 0.4:
            fraction = self.base_fraction * 0.5
        elif ci_width > 0.2:
            fraction = self.base_fraction * 0.75
        else:
            fraction = self.base_fraction
        
        kelly = max(0, (p * (b + 1) - 1) / b)
        return kelly * fraction
```

### Properties

1. **No bankruptcy**: Kelly never bets entire bankroll
2. **Asymptotic optimality**: Maximizes long-run growth rate
3. **Myopic**: One-step lookahead (can be extended to multi-period)
4. **Assumes accurate p**: Sensitive to probability estimation errors

---

## Convergence Theory

### Monte Carlo Convergence

**Definition**: Simulation has converged when additional samples don't significantly change the estimate.

**Window-based test:**

Compare means of two consecutive windows:
```
|μ_window1 - μ_window2| / |μ_window1| < ε
```

Typically ε = 0.001 (0.1% change).

**Implementation:**

```python
class ConvergenceTracker:
    def __init__(self, window_size=100, tolerance=0.001):
        self.window_size = window_size
        self.tolerance = tolerance
        self.results = []
    
    def add_result(self, result):
        self.results.append(result)
    
    def check_convergence(self):
        if len(self.results) < 2 * self.window_size:
            return False
        
        window1 = self.results[-2*self.window_size:-self.window_size]
        window2 = self.results[-self.window_size:]
        
        mean1 = np.mean(window1)
        mean2 = np.mean(window2)
        
        if abs(mean1) < 1e-10:
            return abs(mean2 - mean1) < self.tolerance
        
        relative_diff = abs(mean2 - mean1) / abs(mean1)
        return relative_diff < self.tolerance
```

### Statistical Tests

**Welch's t-test** for comparing two windows:
```
t = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂)
```

If |t| < t_critical, windows are not significantly different (converged).

---

## Variance Reduction

### Motivation

Standard Monte Carlo error: σ/√n

Variance reduction techniques reduce σ, achieving same accuracy with fewer samples.

### Antithetic Variates

For estimating E[f(U)] where U ~ Uniform[0,1]:

Use pairs: (U, 1-U)

**Estimator:**
```
θ̂ = (1/n) Σᵢ [f(Uᵢ) + f(1-Uᵢ)] / 2
```

**Variance reduction:** If f is monotonic, Cov(f(U), f(1-U)) < 0, reducing variance.

### Control Variates

If we know E[Y] exactly and Y correlates with X:
```
X̂_cv = X̂ + c(Y̅ - E[Y])
```

Optimal c = -Cov(X,Y)/Var(Y)

**Variance:**
```
Var[X̂_cv] = Var[X̂](1 - ρ²)
```

where ρ is correlation between X and Y.

### Importance Sampling

Sample from g(x) instead of f(x):
```
E_f[h(X)] = E_g[h(X) × f(X)/g(X)]
```

**Estimator:**
```
θ̂ = (1/n) Σᵢ h(Xᵢ) w(Xᵢ)
```

where w(x) = f(x)/g(x) is importance weight.

**Optimal g:** Proportional to |h(x)|f(x)

### Stratified Sampling

Partition space into strata, sample from each:
```
θ̂_strat = Σₖ wₖ θ̂ₖ
```

where wₖ = P(stratum k), θ̂ₖ = estimate from stratum k.

**Variance:**
```
Var[θ̂_strat] = Σₖ wₖ² σₖ²/nₖ ≤ Var[θ̂_simple]
```

Equality when sampling uniformly, inequality otherwise.

---

## Multi-Armed Bandits

### Problem Formulation

K arms (actions), each with unknown reward distribution. Goal: maximize cumulative reward over T rounds.

**Regret:**
```
R(T) = T × μ* - Σₜ r_t
```

where μ* is the best arm's mean, r_t is reward at time t.

### Algorithms

#### ε-Greedy

With probability ε, explore uniformly. With probability 1-ε, exploit best arm.

**Regret:** O(T^(2/3))

#### Upper Confidence Bound (UCB)

```
choose arg max_k [μ̂_k + √(2 log t / n_k)]
```

Balances exploitation (μ̂_k) and exploration (uncertainty term).

**Regret:** O(√T log T) (optimal)

#### Thompson Sampling

Maintain posterior for each arm. Sample from posteriors, choose highest.

```
For each arm k:
    θ_k ~ posterior_k
choose arg max_k θ_k
```

**For Bernoulli bandits:** Use Beta posteriors.

**Regret:** O(√T log T) (optimal, Bayesian)

**Implementation:**

```python
class ThompsonSampling:
    def __init__(self, n_arms):
        self.estimators = [BetaEstimator(alpha=1.0, beta=1.0) 
                          for _ in range(n_arms)]
    
    def select_arm(self):
        samples = [est.sample(1)[0] for est in self.estimators]
        return np.argmax(samples)
    
    def update(self, arm, reward):
        if reward > 0:
            self.estimators[arm].update(successes=1, failures=0)
        else:
            self.estimators[arm].update(successes=0, failures=1)
```

### Contextual Bandits

Actions depend on context x:
```
choose a_t = arg max_a E[r | x_t, a]
```

**Algorithms:**
- **LinUCB**: Linear models with UCB
- **Neural Bandits**: Deep learning + Thompson Sampling
- **Bayesian Optimization**: Gaussian Processes

---

## References

### Books

1. **Robert & Casella** (2004). *Monte Carlo Statistical Methods*. Springer.
2. **Gelman et al.** (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
3. **Sutton & Barto** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
4. **Lattimore & Szepesvári** (2020). *Bandit Algorithms*. Cambridge University Press.

### Papers

1. **Kelly, J. L.** (1956). "A new interpretation of information rate." *Bell System Technical Journal*, 35(4), 917-926.
2. **Thorp, E. O.** (1969). "Optimal gambling systems for favorable games." *Review of the International Statistical Institute*, 37(3), 273-293.
3. **Thompson, W. R.** (1933). "On the likelihood that one unknown probability exceeds another." *Biometrika*, 25(3/4), 285-294.
4. **Agrawal, S., & Goyal, N.** (2012). "Analysis of Thompson Sampling for the Multi-armed Bandit Problem." *COLT*.

### Standards & Guidelines

1. **PEP 8** - Style Guide for Python Code
2. **PEP 257** - Docstring Conventions
3. **PEP 484** - Type Hints
4. **EPA** (2014). *Risk Assessment Guidelines*
5. **TQMP** - Tutorial on Monte Carlo simulation design practices

---

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| E[X] | Expected value of X |
| Var[X] | Variance of X |
| p, q | Probability (q = 1-p) |
| α, β | Beta distribution parameters |
| μ, σ² | Mean and variance |
| θ | Parameter of interest |
| f*, f_Kelly | Optimal Kelly fraction |
| n, T | Number of samples/trials |
| CI | Confidence/Credible Interval |
| →^(a.s.) | Converges almost surely |
| O(·) | Big-O notation (order of magnitude) |


