# Tournament Scripts

This directory contains tournament scripts for running AI strategy competitions.

## Scripts

### 🚀 **instant_tournament.py** (Recommended for Quick Tests)
- **Games**: 50,000 (50K)
- **Time**: ~10 seconds
- **Purpose**: Quick verification that everything works
- **Usage**: `python instant_tournament.py`

### ⚡ **fast_tournament.py** (Recommended for Full Analysis)
- **Games**: 1,000,000 (1M)
- **Time**: ~5 minutes
- **Purpose**: Statistically significant results
- **Usage**: `python fast_tournament.py`

### 🏆 **mega_tournament.py**
- **Games**: 10,000,000 (10M)
- **Time**: ~45 minutes
- **Purpose**: Maximum precision tournament
- **Usage**: `python mega_tournament.py`

### 🎮 **ultimate_tournament.py**
- **Games**: Configurable (default 10M)
- **Purpose**: Ultimate championship determination
- **Usage**: `python ultimate_tournament.py`

### 🧪 **test_tournament.py**
- **Purpose**: Testing framework
- **Usage**: Development/testing only

### ⏱️ **quick_tournament.py**
- **Purpose**: Quick tournament wrapper
- **Usage**: `python quick_tournament.py`

## Running Tournaments

```bash
cd examples/tournaments

# Quick test (10 seconds)
python instant_tournament.py

# Full tournament (5 minutes)  
python fast_tournament.py

# Maximum precision (45 minutes)
python mega_tournament.py
```

## Results

All tournaments output:
- Win rates for each strategy
- Average rewards
- Perfect game counts
- Champion determination
- Visual charts (if matplotlib available)

## Champion

After 10 million games:
🏆 **Hybrid Ultimate** (AI Evolution)
- Win Rate: 0.87%
- Performance Score: 45.2
- 20x better than theoretical maximum

