# Testing and Validation Procedures

This document provides detailed procedures for testing and validating the Applied Probability and Automation Framework setup.

## 1. Python Backend Integration Test

To verify that all Python modules are correctly imported and integrated, follow these steps:

1.  Navigate to the `python-backend` directory:

    ```bash
    cd applied-probability-framework/python-backend
    ```

2.  Run the `main.py` script:

    ```bash
    python3.11 -m src.main
    ```

3.  **Expected Output:** You should see the following message in your console, indicating that all modules were imported without any errors:

    ```
    Applied Probability and Automation Framework
    All modules imported successfully.
    ```

If you encounter any `ModuleNotFoundError` or other import-related errors, double-check the file paths and ensure that all dependencies from `requirements.txt` are installed in your Python environment.

## 2. Java GUI and Python Backend Connection

To test the connection between the Java GUI and the Python backend, you will need to have both components running simultaneously. This test will require further development to establish a communication protocol (e.g., using sockets or a REST API) between the two.

**Future Steps:**

1.  Implement a server in the Python backend (e.g., using Flask or a similar framework) to expose an API.
2.  In the `GameControlPanel.java` file, add code to make HTTP requests to the Python backend's API.
3.  Run both the Python backend server and the Java GUI to test the connection.

## 3. Interactive Dashboard Data Access

To test the interactive dashboard, follow these steps:

1.  Navigate to the `interactive-dashboard` directory:

    ```bash
    cd applied-probability-framework/interactive-dashboard
    ```

2.  Install the Node.js dependencies:

    ```bash
    npm install
    ```

3.  Start the development server:

    ```bash
    npm start
    ```

4.  Open your web browser and go to `http://localhost:3000` (or the address provided by the development server).

**Expected Outcome:** The React dashboard should render in your browser. To fully test the dashboard, you will need to connect it to a data source, which would typically be the Python backend. This will involve fetching data from the backend's API and displaying it in the dashboard's components.

## 4. Final Configuration Checklist

Before running the full application, it is crucial to verify that all configuration files have been updated with the correct file paths for your local environment. Use the following checklist to ensure everything is set up correctly:

-   [ ] `python-backend/src/drl_config.json`: Verify any file paths if they are added in the future.
-   [ ] `python-backend/src/multi_agent_config.json`: Verify any file paths if they are added in the future.
-   [ ] `python-backend/src/behavioral_config.json`: Verify any file paths if they are added in the future.
-   [ ] `python-backend/test_config.json`: Verify any file paths if they are added in the future.

By following these testing and validation procedures, you can ensure that the Applied Probability and Automation Framework is correctly set up and ready for use.



## 5. K-Fold Backtesting

To ensure the robustness and generalization of your strategies, implement k-fold backtesting. This involves dividing your historical simulation data into `k` subsets, training your models on `k-1` subsets, and validating on the remaining subset. This process is repeated `k` times, with each subset used exactly once for validation.

**Procedure:**

1.  **Data Preparation:** Ensure your `logs/sessions.jsonl` contains sufficient historical data from various game conditions (board sizes, mine counts, etc.).
2.  **Split Data:** Implement a function to split your logged sessions into `k` folds.
3.  **Iterative Training/Validation:** For each fold:
    *   Train/calibrate your strategy (e.g., Bayesian estimators, DRL agents) using data from `k-1` folds.
    *   Run simulations using the strategy on the validation fold.
    *   Record performance metrics (win rate, payout, drawdown, etc.).
4.  **Aggregate Results:** Calculate the average and standard deviation of performance metrics across all `k` folds to get a robust estimate of your strategy's performance.

**Expected Outcome:** A more reliable assessment of strategy performance, less prone to overfitting to a single dataset.

## 6. Stress Tests

Stress testing involves pushing your strategies and simulator to their limits by introducing extreme or unexpected conditions. This helps identify vulnerabilities and ensures stability under adverse scenarios.

**Procedure:**

1.  **Extreme Parameters:** Run simulations with:
    *   Very high/low mine counts.
    *   Unusually large/small board sizes.
    *   High volatility in payouts.
2.  **Adversarial Conditions:** Simulate scenarios where:
    *   The game's underlying probabilities subtly shift over time (model drift).
    *   Opponent strategies (if applicable) become highly optimized.
    *   Latency or network issues are introduced (for live system considerations).
3.  **Resource Limits:** Monitor resource usage (CPU, memory) during prolonged, high-intensity simulations to ensure the framework can handle scale.

**Expected Outcome:** Identification of breakpoints, performance bottlenecks, and areas where strategies might fail under extreme conditions. This will inform further improvements and make your framework more resilient, like a true shonen hero facing their ultimate foe!

