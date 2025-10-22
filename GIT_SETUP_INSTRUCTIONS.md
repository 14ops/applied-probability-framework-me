# Git Setup and GitHub Update Instructions

## Step 1: Install Git (if not already installed)

### Download Git for Windows
1. Go to: https://git-scm.com/download/win
2. Download and install Git for Windows
3. During installation, accept the default options
4. Restart your PowerShell/Command Prompt after installation

### Verify Installation
```bash
git --version
```

## Step 2: Configure Git (First Time Setup)

If this is your first time using Git, configure your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Initialize Repository (if not already done)

If this isn't already a Git repository:

```bash
cd C:\Users\Seth\applied-probability-framework-me-main
git init
```

## Step 4: Add Remote Repository

If you haven't connected to GitHub yet:

```bash
# Replace with your actual GitHub repository URL
git remote add origin https://github.com/YOUR_USERNAME/applied-probability-framework.git
```

Check your remotes:
```bash
git remote -v
```

## Step 5: Commit and Push Changes

### Option A: Use the Batch Script (Easiest)

Simply double-click `update_github.bat` or run:
```bash
update_github.bat
```

### Option B: Manual Commands

```bash
# 1. Check status
git status

# 2. Add all files
git add .

# 3. Commit with message
git commit -m "feat: transform framework to professional-grade (v1.0)"

# 4. Push to GitHub
git push -u origin main
```

Note: Use `master` instead of `main` if your default branch is named `master`.

## Step 6: Verify on GitHub

1. Go to your GitHub repository
2. Refresh the page
3. You should see all the new files and changes!

## Troubleshooting

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin YOUR_REPO_URL
```

### Error: "failed to push some refs"
```bash
# Pull and merge first
git pull origin main --rebase
git push origin main
```

### Error: "Permission denied (publickey)"
You need to set up SSH keys or use HTTPS with a personal access token.

**For HTTPS:**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token with 'repo' permissions
3. Use the token as your password when pushing

**For SSH:**
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your.email@example.com"`
2. Add to GitHub: Settings → SSH and GPG keys → New SSH key
3. Use SSH URL: `git@github.com:USERNAME/REPO.git`

### Create GitHub Repository First

If you don't have a GitHub repository yet:

1. Go to https://github.com/new
2. Repository name: `applied-probability-framework`
3. Description: "Professional Monte Carlo simulation framework"
4. Set to Public or Private
5. Don't initialize with README (we already have one)
6. Click "Create repository"
7. Copy the repository URL and use it in Step 4 above

## What Will Be Committed?

All the new professional framework features:

### New Core Modules
- `src/python/core/` - Complete rewrite with abstract base classes
- `src/python/cli.py` - Professional CLI
- `src/python/tests/` - Comprehensive test suite (100+ tests)

### Documentation
- `README.md` - Professional overview
- `docs/API_REFERENCE.md` - Complete API docs
- `docs/TUTORIALS.md` - Step-by-step guides
- `docs/THEORETICAL_BACKGROUND.md` - Math foundations
- `docs/IMPROVEMENTS_SUMMARY.md` - Detailed improvements
- `QUICKSTART.md` - 5-minute quick start
- `CONTRIBUTING.md` - Contribution guidelines

### CI/CD
- `.github/workflows/ci.yml` - Automated testing
- `.github/workflows/release.yml` - Release automation
- `pyproject.toml` - Build configuration
- `setup.cfg` - Tool configuration

### Other
- `LICENSE` - MIT License
- Updated `requirements.txt`

## Files You May Want to .gitignore

Create a `.gitignore` file to exclude:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Framework specific
results/
experiments/
logs/
*.jsonl
```

## Summary of Commands

Once Git is installed and configured:

```bash
# Navigate to project
cd C:\Users\Seth\applied-probability-framework-me-main

# Check status
git status

# Add all changes
git add .

# Commit
git commit -m "feat: transform framework to professional-grade (v1.0)"

# Push to GitHub
git push -u origin main
```

---

**Need Help?** 
- Git documentation: https://git-scm.com/doc
- GitHub guides: https://guides.github.com/
- Stack Overflow: https://stackoverflow.com/questions/tagged/git

