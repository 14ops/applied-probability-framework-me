# Applied Probability and Automation Framework for High-RTP Games

**A Research + Engineering Hybrid Project**

This repository contains the complete source code, documentation, and research materials for a sophisticated framework designed to analyze and automate betting strategies in high-Return-to-Player (RTP) games. The project integrates theoretical probability analysis with practical software engineering to create an adaptive, multi-component system.

---

## 🚀 Project Overview

The framework is composed of three main components, each housed in a dedicated directory:

1.  **Backend (Python):** The analytical engine for game simulation, strategy execution, and data analysis.
2.  **Frontend (Interactive Dashboard):** A modern web-based interface for configuration, control, and real-time visualization of results.
3.  **GUI (Java):** A legacy Java-based control panel, included for historical and academic completeness.

## 📁 Repository Structure

| Directory | Component | Description |
| :--- | :--- | :--- |
| `backend/python` | Python Engine | Core logic for simulation, automation, and advanced strategies. |
| `frontend/dashboard` | Interactive Dashboard | React-based web interface for control and visualization. |
| `gui/java` | Java GUI | Legacy Java Swing/FX control panel. |
| `infrastructure/supabase` | Database/Infra | Supabase migration files and infrastructure setup. |
| `docs/` | Documentation | Comprehensive project documentation and research papers. |
| `docs/papers` | Research Papers | Original PDF documents and academic reports. |
| `docs/guides` | Implementation Guides | Detailed guides on setup, testing, and project operation. |
| `docs/strategies` | Strategy Details | In-depth explanations of the four core strategies and their formulas. |
| `docs/ARCHIVE` | Archived Documents | Older or superseded README versions for historical reference. |

## 📚 Key Documentation

| Document | Location | Description |
| :--- | :--- | :--- |
| **Main README** | `/README.md` (You are here) | High-level overview and repository map. |
| **Implementation Guide** | `docs/guides/implementation_guide.md` | Step-by-step instructions for setting up and running the framework. |
| **Testing & Validation** | `docs/guides/testing_and_validation.md` | Details on the simulation environment and performance metrics. |
| **Strategy Formulas** | `docs/strategies/strategy_formulas.md` | Mathematical formulas and logical representations for the core strategies. |
| **Dashboard README** | `frontend/dashboard/README.md` | Specific instructions for the interactive web dashboard. |

## 🛠️ Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/14ops/applied-probability-framework-shihab-belal.git
    cd applied-probability-framework-shihab-belal
    ```
2.  **Set up the Python Backend:**
    ```bash
    cd backend/python
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # Quick smoke test (optional deps are handled gracefully)
    python -c "import sys; sys.path.insert(0, '$(pwd)'); from src.main import main; main()"
    ```
3.  **Set up the Interactive Dashboard:**
    ```bash
    cd frontend/dashboard
    npm install
    npm run dev
    # See frontend/dashboard/README.md for more details
    ```
4.  **Set up the Java GUI:**
    ```bash
    cd gui/java
    ./compile_and_run.sh
    ```

---


