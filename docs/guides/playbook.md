# Applied Probability and Automation Framework: Import and Setup Playbook

This playbook provides a comprehensive guide for importing and setting up the Applied Probability and Automation Framework. Following these steps will ensure that all dependencies, modules, and files are properly integrated for seamless operation.

## 1. Project Structure Setup

The project is organized into the following directory structure:

```
applied-probability-framework/
├── python-backend/
│   ├── src/
│   ├── visualizations/
│   └── requirements.txt
├── java-gui/
├── interactive-dashboard/
├── README.md
└── ... (other files)
```

This structure separates the Python backend, Java GUI, and interactive dashboard for clarity and maintainability.

## 2. Python Backend Dependencies

To install the required Python packages, navigate to the `python-backend` directory and run the following command:

```bash
pip install -r requirements.txt
```

This will install all necessary libraries, including NumPy, SciPy, Pandas, Scikit-learn, Matplotlib, TensorFlow, pgmpy, and Plotly.

## 3. Core Python Modules Import

Ensure that all Python files include the necessary import statements at the top. For example, `main.py` might start with:

```python
import numpy as np
from src import game_simulator
from src import strategies
from src import utils
```

## 4. Advanced Strategy Modules

The following advanced strategy modules are included in the `python-backend/src` directory:

- `advanced_strategies.py`
- `rintaro_okabe_strategy.py`
- `monte_carlo_tree_search.py`
- `markov_decision_process.py`
- `strategy_auto_evolution.py`
- `confidence_weighted_ensembles.py`

## 5. AI and Machine Learning Components

The AI and ML components are located in the `python-backend/src` directory and include:

- **Reinforcement Learning:** `drl_environment.py`, `drl_agent.py`, `drl_config.json`
- **Bayesian Inference:** `bayesian_mines.py`
- **Adversarial Training:** `human_data_collector.py`, `adversarial_agent.py`, `adversarial_detector.py`, `adversarial_trainer.py`

## 6. Multi-Agent System Components

The multi-agent system components are in the `python-backend/src` directory:

- `multi_agent_core.py`
- `agent_comms.py`
- `multi_agent_simulator.py`
- `multi_agent_config.json`

## 7. Behavioral Economics Integration

The behavioral economics modules are in the `python-backend/src` directory:

- `behavioral_value.py`
- `behavioral_probability.py`
- `behavioral_config.json`

## 8. Visualization and Analytics

The visualization modules are in the `python-backend/visualizations` directory:

- `visualization.py`
- `advanced_visualizations.py`
- `comprehensive_visualizations.py`
- `specialized_visualizations.py`
- `realtime_heatmaps.py`

## 9. Java GUI Components

The Java GUI files are in the `java-gui` directory. To compile and run the GUI, execute the `compile_and_run.sh` script:

```bash
cd java-gui
./compile_and_run.sh
```

## 10. Interactive Dashboard Setup

The interactive dashboard files are in the `interactive-dashboard` directory. To set it up, navigate to the directory and install the Node.js dependencies:

```bash
cd interactive-dashboard
npm install
```

Then, you can start the development server:

```bash
npm start
```

## 11. Configuration and Testing

Configuration and testing files are located in the `python-backend` directory:

- `test_config.json`
- `drl_evaluation.py`
- `ab_testing_framework.py`

## 12. Documentation and Reports

Project documentation and reports are in the root directory:

- `README.md`
- `README_ENHANCED.md`
- `README_FINAL.md`
- `character_strategies_explained.md`
- `strategy_formulas.md`

## 13. Integration Testing

To verify that all components are correctly imported and integrated, run the `main.py` file from the `python-backend` directory:

```bash
cd python-backend
python src/main.py
```

Check for any import errors or missing dependencies in the console output.

## 14. Final Configuration

Finally, update all file paths in the configuration files (`drl_config.json`, `multi_agent_config.json`, `behavioral_config.json`, `test_config.json`) to match your local setup. Ensure that the Java GUI can connect to the Python backend and that the React dashboard can access the necessary data.

---

This comprehensive import and setup process will ensure that all components of the Applied Probability and Automation Framework are properly integrated, allowing for seamless operation of the entire system.

