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

## 3. Status of Implementation

Completed actions:
- Created standardized directories: `backend`, `frontend`, `gui`, `infrastructure`, and docs subfolders.
- Moved Python backend to `backend/python` and flattened nested `python-backend`.
- Flattened GUI under `gui/java` (removed `java-gui` nesting).
- Consolidated dashboard under `frontend/dashboard` and removed deprecated `interactive-dashboard/` subapp.
- Flattened `infrastructure/supabase` (moved nested `supabase/` up one level).

Remaining considerations:
- Verify end-to-end integration (optional ML/visualization libs require `pip install -r backend/python/requirements.txt`).
- Ensure Supabase env vars are set for the dashboard (`VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`).

