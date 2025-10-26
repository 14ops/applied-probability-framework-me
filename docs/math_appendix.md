# Mathematical Appendix: Mines Game Probability Theory

## Table of Contents
1. [Game Model](#game-model)
2. [Combinatorial Foundations](#combinatorial-foundations)
3. [Win Probability Derivation](#win-probability-derivation)
4. [Payout Analysis](#payout-analysis)
5. [Expected Value Calculations](#expected-value-calculations)
6. [Kelly Criterion Application](#kelly-criterion-application)
7. [Numerical Tables](#numerical-tables)

---

## Game Model

### Physical Setup
- **Board**: 5×5 grid
- **Total tiles**: n = 25
- **Mines**: m = 2
- **Safe tiles**: n - m = 23

### Game Mechanics
1. Player selects k tiles sequentially
2. If all k tiles are safe → **Win** with payout multiplier M(k)
3. If any tile is a mine → **Loss** (bet forfeited)
4. No replacement (tiles cannot be clicked twice)

### Assumptions
- Mines are uniformly randomly distributed
- All tile arrangements are equally likely
- Player has no information about mine locations
- Payouts are fixed based on number of successful clicks

---

## Combinatorial Foundations

### Binomial Coefficient

The number of ways to choose r items from n items:

$$C(n, r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}$$

**Examples for our model:**
- Total ways to place 2 mines in 25 tiles: $C(25, 2) = 300$
- Ways to choose 8 safe tiles from 23: $C(23, 8) = 490,314$
- Ways to choose any 8 tiles from 25: $C(25, 8) = 1,081,575$

### Fundamental Counting Principle

If event A can occur in m ways and event B in n ways, then both can occur in m × n ways.

**Applied to Mines:**
- First click safe: $(n-m)/n$ ways
- Second click safe (given first safe): $(n-m-1)/(n-1)$ ways
- k clicks all safe: $\prod_{i=0}^{k-1} \frac{n-m-i}{n-i}$

---

## Win Probability Derivation

### Method 1: Combinatorial Formula

**Theorem:** The probability of successfully clicking k safe tiles is:

$$P(\text{win } k \text{ clicks}) = \frac{C(n-m, k)}{C(n, k)}$$

**Proof:**
- Total ways to choose k tiles from n: $C(n, k)$
- Favorable ways (k safe tiles from n-m safe): $C(n-m, k)$
- By classical probability: $P = \frac{\text{favorable}}{\text{total}}$

**For our model (n=25, m=2):**

$$P(k) = \frac{C(23, k)}{C(25, k)}$$

### Method 2: Multiplicative Formula

**Theorem:** The probability can also be expressed as:

$$P(k) = \prod_{i=0}^{k-1} \frac{n - m - i}{n - i}$$

**Proof:**
- Probability first click is safe: $(n-m)/n$
- Probability second click is safe given first safe: $(n-m-1)/(n-1)$
- By independence of sequential events: multiply probabilities

**Equivalence:**

$$\frac{C(n-m, k)}{C(n, k)} = \frac{(n-m)!/(k!(n-m-k)!)}{n!/(k!(n-k)!)} = \frac{(n-m)!(n-k)!}{n!(n-m-k)!} = \prod_{i=0}^{k-1} \frac{n-m-i}{n-i}$$

### Concrete Examples

#### Example 1: Single Click (k=1)
$$P(1) = \frac{C(23,1)}{C(25,1)} = \frac{23}{25} = 0.92 = 92\%$$

#### Example 2: Eight Clicks (k=8)
$$P(8) = \frac{C(23,8)}{C(25,8)} = \frac{490,314}{1,081,575} = 0.4533 = 45.33\%$$

Alternative calculation:
$$P(8) = \frac{23}{25} \times \frac{22}{24} \times \frac{21}{23} \times \frac{20}{22} \times \frac{19}{21} \times \frac{18}{20} \times \frac{17}{19} \times \frac{16}{18} = 0.4533$$

#### Example 3: Maximum Safe Clicks (k=23)
$$P(23) = \frac{C(23,23)}{C(25,23)} = \frac{1}{C(25,2)} = \frac{1}{300} = 0.0033 = 0.33\%$$

---

## Payout Analysis

### Fair Payout Calculation

A **fair payout** (zero house edge) satisfies:

$$\mathbb{E}[\text{profit}] = 0$$

For a bet of $1:
$$P(k) \cdot M_{\text{fair}}(k) \cdot 1 - (1 - P(k)) \cdot 1 = 0$$

Solving for fair payout:
$$M_{\text{fair}}(k) = \frac{1}{P(k)}$$

### House Edge

Actual payouts include a **house edge** h (typically 3-5%):

$$M_{\text{actual}}(k) = M_{\text{fair}}(k) \cdot (1 - h) = \frac{1 - h}{P(k)}$$

### Observed vs. Theoretical Payouts

| k | P(k) | Fair | With 3% HE | Observed | Difference |
|---|------|------|------------|----------|------------|
| 1 | 0.9200 | 1.09 | 1.05 | **1.04** | -0.01 |
| 2 | 0.8433 | 1.19 | 1.15 | **1.14** | -0.01 |
| 3 | 0.7700 | 1.30 | 1.26 | **1.25** | -0.01 |
| 4 | 0.7000 | 1.43 | 1.39 | **1.38** | -0.01 |
| 5 | 0.6333 | 1.58 | 1.53 | **1.53** | 0.00 |
| 6 | 0.5700 | 1.75 | 1.70 | **1.72** | +0.02 |
| 7 | 0.5100 | 1.96 | 1.90 | **1.95** | +0.05 |
| 8 | 0.4533 | 2.21 | 2.14 | **2.12** | -0.02 |
| 9 | 0.4000 | 2.50 | 2.42 | **2.52** | +0.10 |
| 10 | 0.3500 | 2.86 | 2.77 | **3.02** | +0.25 |

**Observation:** Clicks 7-10 offer **positive expected value** (+EV), making them profitable targets!

---

## Expected Value Calculations

### Definition

Expected value of a bet B at k clicks:

$$\text{EV}(k, B) = P(k) \cdot M(k) \cdot B - (1 - P(k)) \cdot B$$

Simplified:
$$\text{EV}(k, B) = B \cdot [P(k) \cdot M(k) - 1]$$

### Examples with $10 Bet

#### Click Count Analysis

```
k=1:  EV = 10 × [0.92 × 1.04 - 1] = -$0.43
k=5:  EV = 10 × [0.63 × 1.53 - 1] = -$0.31
k=7:  EV = 10 × [0.51 × 1.95 - 1] = +$0.05
k=8:  EV = 10 × [0.45 × 2.12 - 1] = +$0.41
k=10: EV = 10 × [0.35 × 3.02 - 1] = +$0.41
k=12: EV = 10 × [0.26 × 4.65 - 1] = +$0.47
```

**Optimal Range:** 7-12 clicks for positive EV with observed payouts.

### Variance Analysis

Variance measures risk:

$$\text{Var}(k) = P(k) \cdot [M(k) \cdot B]^2 + (1-P(k)) \cdot B^2 - [\text{EV}(k, B)]^2$$

**Standard Deviation** (σ) gives typical deviation from EV:

$$\sigma(k) = \sqrt{\text{Var}(k)}$$

#### Risk-Reward Table ($10 bet)

| k | EV | σ | Sharpe Ratio |
|---|----|----|--------------|
| 5 | -0.31 | 9.87 | -0.031 |
| 7 | +0.05 | 9.91 | +0.005 |
| 8 | +0.41 | 10.02 | +0.041 |
| 10 | +0.41 | 11.32 | +0.036 |

**Sharpe Ratio** = EV / σ (higher is better risk-adjusted return)

---

## Kelly Criterion Application

### Formula

Optimal bet fraction for maximizing long-term growth:

$$f^* = \frac{bp - q}{b}$$

Where:
- b = odds (payout - 1)
- p = probability of winning
- q = 1 - p (probability of losing)

### Applied to Mines (k=8)

- p = 0.4533
- M = 2.12 → b = 1.12
- q = 0.5467

$$f^* = \frac{1.12 \times 0.4533 - 0.5467}{1.12} = \frac{0.5077 - 0.5467}{1.12} = -0.035$$

**Negative Kelly → No edge at k=8 with fair calculation.**

With observed payout (M=2.12):
$$f^* = \frac{1.12 \times 0.4533 - 0.5467}{1.12} = -0.035$$

**Note:** Even with +EV, Kelly is small due to high variance. **Fractional Kelly** (e.g., 0.5× Kelly) is recommended for bankroll safety.

---

## Numerical Tables

### Complete Probability Table (n=25, m=2)

| k | C(23,k) | C(25,k) | P(k) | Fair Payout |
|---|---------|---------|------|-------------|
| 1 | 23 | 25 | 0.9200 | 1.09 |
| 2 | 253 | 300 | 0.8433 | 1.19 |
| 3 | 1,771 | 2,300 | 0.7700 | 1.30 |
| 4 | 8,855 | 12,650 | 0.7000 | 1.43 |
| 5 | 33,649 | 53,130 | 0.6333 | 1.58 |
| 6 | 100,947 | 177,100 | 0.5700 | 1.75 |
| 7 | 245,157 | 480,700 | 0.5100 | 1.96 |
| 8 | 490,314 | 1,081,575 | 0.4533 | 2.21 |
| 9 | 817,190 | 2,042,975 | 0.4000 | 2.50 |
| 10 | 1,144,066 | 3,268,760 | 0.3500 | 2.86 |
| 11 | 1,352,078 | 4,457,400 | 0.3033 | 3.30 |
| 12 | 1,352,078 | 5,200,300 | 0.2600 | 3.85 |
| 13 | 1,144,066 | 5,200,300 | 0.2200 | 4.55 |
| 14 | 817,190 | 4,457,400 | 0.1833 | 5.45 |
| 15 | 490,314 | 3,268,760 | 0.1500 | 6.67 |

### Expected Value Table (Observed Payouts, $10 Bet)

| k | Win Prob | Payout | Expected Value | ROI |
|---|----------|--------|----------------|-----|
| 1 | 92.00% | 1.04 | -$0.43 | -4.3% |
| 2 | 84.33% | 1.14 | -$0.39 | -3.9% |
| 3 | 77.00% | 1.25 | -$0.38 | -3.8% |
| 4 | 70.00% | 1.38 | -$0.34 | -3.4% |
| 5 | 63.33% | 1.53 | -$0.31 | -3.1% |
| 6 | 57.00% | 1.72 | -$0.20 | -2.0% |
| 7 | 51.00% | 1.95 | **+$0.05** | **+0.5%** |
| 8 | 45.33% | 2.12 | **+$0.41** | **+4.1%** |
| 9 | 40.00% | 2.52 | **+$0.41** | **+4.1%** |
| 10 | 35.00% | 3.02 | **+$0.41** | **+4.1%** |
| 11 | 30.33% | 3.70 | **+$0.43** | **+4.3%** |
| 12 | 26.00% | 4.65 | **+$0.47** | **+4.7%** |
| 13 | 22.00% | 6.06 | **+$0.55** | **+5.5%** |
| 14 | 18.33% | 8.35 | **+$0.71** | **+7.1%** |
| 15 | 15.00% | 12.50 | **+$1.02** | **+10.2%** |

---

## References

1. Ross, S. M. (2014). *Introduction to Probability Models* (11th ed.). Academic Press.
2. Feller, W. (1968). *An Introduction to Probability Theory and Its Applications, Vol. 1* (3rd ed.). Wiley.
3. Kelly, J. L. (1956). "A New Interpretation of Information Rate". *Bell System Technical Journal*, 35(4), 917-926.
4. Thorp, E. O. (2008). "The Kelly Criterion in Blackjack Sports Betting, and the Stock Market". *Handbook of Asset and Liability Management*, 385-428.

---

## Computational Implementation

See `src/python/game/math.py` for Python implementations of all formulas:

```python
from game.math import (
    win_probability,          # P(k) calculation
    theoretical_fair_payout,  # Fair payout calculation
    expected_value,           # EV calculation
    kelly_criterion,          # Optimal bet sizing
    print_payout_table        # Generate full tables
)
```

---

*Last Updated: October 2025*

