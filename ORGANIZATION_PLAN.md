# Repository Organization Plan

The goal is to clean up the root directory, consolidate documentation, standardize the project structure, and improve maintainability.

## 1. Documentation Consolidation and Cleanup

The root directory is cluttered with multiple README and documentation files.

| File | Proposed Action | New Location/Status | Rationale |
| :--- | :--- | :--- | :--- |
| `README.md` | **Keep and Refine** | `/README.md` | Standard entry point. Will be updated to be a high-level overview and a table of contents. |
| `README_FINAL.md` | **Merge/Archive** | `/docs/ARCHIVE/README_FINAL.md` | Likely the best content. Will be merged into the main `README.md` and then archived. |
| `README_ENHANCED.md` | **Archive** | `/docs/ARCHIVE/README_ENHANCED.md` | Archive for historical reference. |
| `APP_README.md` | **Move/Rename** | `/interactive-dashboard/README.md` | Specific to the dashboard application. |
| `character_strategies_explained.md` | **Move** | `/docs/strategies/character_strategies_explained.md` | Detailed documentation. |
| `implementation_guide.md` | **Move** | `/docs/guides/implementation_guide.md` | Detailed documentation. |
| `strategy_formulas.md` | **Move** | `/docs/strategies/strategy_formulas.md` | Detailed documentation. |
| `testing_and_validation.md` | **Move** | `/docs/guides/testing_and_validation.md` | Detailed documentation. |
| `playbook.md` | **Move** | `/docs/guides/playbook.md` | Detailed documentation. |
| `/docs` folder content | **Keep** | `/docs/papers/` | Rename to `papers` to better reflect the content (PDFs). |

## 2. Standardizing Project Structure

The project has three main components: a Python backend, a Java GUI, and a web-based interactive dashboard.

| Component | Current Location | Proposed Location | Rationale |
| :--- | :--- | :--- | :--- |
| **Python Backend** | `/python-backend` | `/backend/python` | Standardize to a `backend` directory. |
| **Java GUI** | `/java-gui` | `/gui/java` | Standardize to a `gui` directory. |
| **Interactive Dashboard** | `/interactive-dashboard` | `/frontend/dashboard` | Standardize to a `frontend` directory. |
| **Web Root Files** | `/index.html`, `/src`, `/package.json`, etc. | `/frontend/dashboard` | Consolidate all web-related files into the dashboard directory. |
| **Supabase** | `/supabase` | `/infrastructure/supabase` | Group infrastructure-related files. |

## 3. Implementation Steps

1.  **Create new directories:** `backend`, `frontend`, `gui`, `infrastructure`, `docs/ARCHIVE`, `docs/guides`, `docs/strategies`, `docs/papers`.
2.  **Move and Rename Components:**
    *   Move `/python-backend` to `/backend/python`.
    *   Move `/java-gui` to `/gui/java`.
    *   Move `/interactive-dashboard` to `/frontend/dashboard`.
    *   Move `/supabase` to `/infrastructure/supabase`.
3.  **Consolidate Web Files:** Move root-level web files (`index.html`, `src`, `package.json`, `package-lock.json`, `vite.config.js`) into `/frontend/dashboard`.
4.  **Consolidate Documentation:**
    *   Move `README_FINAL.md` and `README_ENHANCED.md` to `/docs/ARCHIVE`.
    *   Move `APP_README.md` to `/frontend/dashboard/README.md`.
    *   Move `character_strategies_explained.md` and `strategy_formulas.md` to `/docs/strategies`.
    *   Move `implementation_guide.md`, `testing_and_validation.md`, and `playbook.md` to `/docs/guides`.
    *   Move all files from `/docs` to `/docs/papers`.
5.  **Update Main README:** Write a new, concise `README.md` that serves as a project overview and table of contents, linking to the new documentation structure.
6.  **Cleanup:** Remove all original files from the root directory.
7.  **Final Review and Commit.**

