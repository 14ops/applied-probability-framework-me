# How to Use the Applied Probability Framework Executables

You now have **TWO versions** of the Applied Probability Framework:

## 🖥️ GUI Version (Recommended for Most Users)

**Location:** `dist\AppliedProbabilityFrameworkGUI\AppliedProbabilityFrameworkGUI.exe`

**How to Use:**
1. **Double-click** `AppliedProbabilityFrameworkGUI.exe` to open
2. A window will appear with:
   - **Simulator dropdown** - Select "mines" (currently only option)
   - **Strategy dropdown** - Choose from: conservative, basic, random, or aggressive
   - **Simulations** - Enter how many simulations to run (default: 1000)
   - **Run in Parallel** - Check to use all CPU cores for faster execution
3. Click **▶ Run Simulation** to start
4. Watch the progress bar and results appear in real-time
5. Click **💾 Save Results** to export results to JSON

**Features:**
- ✅ Windows you can see and interact with
- ✅ Easy to use - just point and click
- ✅ Progress bars and live updates
- ✅ Save results to files
- ✅ No command-line knowledge needed

---

## ⌨️ CLI Version (Command-Line Interface)

**Location:** `dist\AppliedProbabilityFramework\AppliedProbabilityFramework.exe`

**How to Use:**
1. Open **PowerShell** or **Command Prompt**
2. Navigate to the folder:
   ```powershell
   cd dist\AppliedProbabilityFramework
   ```
3. Run commands:
   ```powershell
   # See all options
   .\AppliedProbabilityFramework.exe --help
   
   # Run 1000 simulations (default)
   .\AppliedProbabilityFramework.exe run
   
   # Run custom number of simulations
   .\AppliedProbabilityFramework.exe run --num-simulations 5000
   
   # Use different strategy
   .\AppliedProbabilityFramework.exe run --strategy aggressive
   
   # Run in parallel
   .\AppliedProbabilityFramework.exe run --parallel --jobs 8
   
   # List available plugins
   .\AppliedProbabilityFramework.exe plugins --list
   ```

**Features:**
- ✅ Fast and lightweight
- ✅ Scriptable and automatable
- ✅ Perfect for batch processing
- ✅ Can be called from other programs
- ✅ Advanced options available

---

## 📁 Distribution

To share with others:

### GUI Version
Share the entire `dist\AppliedProbabilityFrameworkGUI\` folder containing:
- `AppliedProbabilityFrameworkGUI.exe`
- `_internal\` folder (required dependencies)

### CLI Version
Share the entire `dist\AppliedProbabilityFramework\` folder containing:
- `AppliedProbabilityFramework.exe`
- `_internal\` folder (required dependencies)

**Important:** Users do NOT need Python installed - these are standalone executables!

---

## 🎮 Available Strategies

- **conservative** - Cautious play, uses Bayesian estimation
- **basic** - Similar to conservative
- **random** - Random cell selection
- **aggressive** - More risk-taking behavior

---

## 📊 Understanding Results

**Mean Reward:** Average outcome across all simulations (-1.0 = all losses, positive = wins)

**Success Rate:** Percentage of games won

**95% CI:** Confidence interval for the mean reward

**Execution Time:** How long the simulations took

---

## ⚠️ Note About Game Difficulty

The Mines game is intentionally challenging:
- 5x5 board with 3 mines = 12% mine density
- Most strategies will have low success rates (0-10%)
- This is expected behavior!

The framework is working correctly - the game is just difficult!

---

## 🚀 Quick Start

**Easiest way:** Double-click `AppliedProbabilityFrameworkGUI.exe` and click "Run Simulation"!

Enjoy! 🎉

