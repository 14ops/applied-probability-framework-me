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

