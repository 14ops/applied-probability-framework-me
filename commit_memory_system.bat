@echo off
echo Committing AI Memory System...

REM Add the files
git add src\python\core\ai_memory_system.py
git add scripts\initialize_ai_memory.py
git add scripts\verify_ai_facts.py
git add scripts\README.md
git add docs\AI_MEMORY_SYSTEM.md
git add .gitignore

REM Show status
git status

REM Commit
git commit -m "feat: Add AI Memory System - Prevents AI hallucinations through persistent fact verification"

REM Push
git push origin main

echo Done!
pause

