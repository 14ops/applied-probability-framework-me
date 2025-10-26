# AI Memory System - Preventing Hallucinations

## Overview

The AI Memory System maintains a persistent, verified record of project state to prevent hallucinations. Instead of relying on potentially outdated or incorrect information, the AI can verify facts against stored, ground-truth data.

## Key Features

### 🧠 Persistent Memory
- JSON-based storage of verified facts
- Survives across sessions
- Version controlled with project

### 🔍 Fact Verification
- Check any claim against stored data
- Confidence levels for all facts
- Automatic staleness detection

### 📊 Project State Tracking
- File structure and locations
- Code statistics and metrics
- Git commit history
- Test results
- Configuration values

### 📚 Documentation Registry
- All documentation indexed
- Searchable by keywords
- Category-based organization

### 🔄 Automatic Updates
- Scan project structure
- Update statistics
- Record changes
- Track versions

## Files

### Core System
- **`src/python/core/ai_memory_system.py`** - Main memory system
- **`ai_memory.json`** - Persistent storage (git-ignored)

### Scripts
- **`scripts/initialize_ai_memory.py`** - Initialize memory with current state
- **`scripts/verify_ai_facts.py`** - Verify facts interactively

## Usage

### Initialize Memory

```bash
# Scan project and store current state
python scripts/initialize_ai_memory.py
```

This creates `ai_memory.json` with:
- ✅ Complete file structure
- ✅ Code statistics
- ✅ Git state
- ✅ Verified facts
- ✅ Documentation index
- ✅ Configuration values

### Verify Facts

```bash
# Check facts interactively
python scripts/verify_ai_facts.py
```

### In Code

```python
from core.ai_memory_system import get_memory

# Get memory instance
memory = get_memory()

# Store a fact
memory.store_fact('project', 'champion', 'Hybrid Ultimate')

# Retrieve a fact
champion = memory.get_fact('project', 'champion')

# Verify a fact
is_correct = memory.verify_fact('project', 'champion', 'Hybrid Ultimate')

# Check file exists
exists = memory.verify_path_exists('docs/evolution/QUICKSTART.md')

# Find documentation
docs = memory.find_documentation('evolution')
```

## Stored Information

### Project Structure
```python
memory.memory['structure']
{
    'directories': { ... },
    'files': { ... },
    'last_scanned': '2024-10-26T...',
}
```

### Code Statistics
```python
memory.memory['code_stats']
{
    'total_lines': 8500,
    'python_files': 75,
    'test_files': 20,
    'last_updated': '2024-10-26T...',
}
```

### Verified Facts
```python
memory.memory['facts']['verified']
{
    'project': {
        'name': { 'value': 'Applied Probability Framework', ... },
        'repository': { 'value': 'https://...', ... },
    },
    'tournament': {
        'champion_strategy': { 'value': 'Hybrid Ultimate', ... },
        'champion_win_rate': { 'value': 0.87, ... },
        'total_games_analyzed': { 'value': 10000000, ... },
    },
    'strategies': { ... },
    'locations': { ... },
}
```

### Git History
```python
memory.memory['git']
{
    'branch': 'main',
    'last_commit': { ... },
    'commits': [ ... ],
}
```

### Documentation
```python
memory.memory['documentation']
{
    'docs/evolution/QUICKSTART.md': {
        'title': 'Evolution Quick Start',
        'description': '5-minute guide...',
        'category': 'evolution',
        'timestamp': '2024-10-26T...',
    },
    ...
}
```

## API Reference

### Initialization
```python
from core.ai_memory_system import AIMemorySystem

memory = AIMemorySystem("ai_memory.json")
```

### Project Structure
```python
# Scan and store structure
stats = memory.scan_project_structure(".")

# Check if path exists
exists = memory.verify_path_exists("some/file.py")

# Find file location
location = memory.get_file_location("quickstart.md")
```

### Code Statistics
```python
# Update statistics
stats = memory.update_code_stats(".")

# Get current stats
stats = memory.get_code_stats()
```

### Fact Management
```python
# Store verified fact
memory.store_fact('category', 'key', value, 'verification_method')

# Retrieve fact
value = memory.get_fact('category', 'key')

# Verify fact
is_correct = memory.verify_fact('category', 'key', expected_value)

# Deprecate fact
memory.deprecate_fact('category', 'key', 'reason')
```

### Git Tracking
```python
# Record commit
memory.record_git_commit(hash, message, files_changed)

# Get last commit
commit = memory.get_last_commit()

# Get history
history = memory.get_commit_history(limit=10)
```

### Configuration
```python
# Store config
memory.store_config('name', config_dict)

# Retrieve config
config = memory.get_config('name')
```

### Documentation
```python
# Register documentation
memory.register_documentation(path, title, description, category)

# Find documentation
results = memory.find_documentation('search_term')
```

### Test Results
```python
# Record test run
memory.record_test_run(passed, failed, total, coverage, duration)

# Get latest results
results = memory.get_latest_test_results()
```

### Utilities
```python
# Get memory summary
summary = memory.get_memory_summary()

# Export memory
memory.export_memory('backup.json')

# Import memory
memory.import_memory('backup.json')

# Save changes
memory.save()
```

## Preventing Hallucinations

### Before Making a Claim

```python
# ❌ BAD: Just state it
print("The champion is Hybrid Ultimate")

# ✅ GOOD: Verify first
champion = memory.get_fact('tournament', 'champion_strategy')
if champion:
    print(f"The champion is {champion}")
else:
    print("Champion information not available in memory")
```

### When Citing Statistics

```python
# ❌ BAD: Use potentially wrong numbers
print("Win rate is 0.87%")

# ✅ GOOD: Get from memory
win_rate = memory.get_fact('tournament', 'champion_win_rate')
if win_rate:
    print(f"Win rate is {win_rate}%")
```

### When Referencing Files

```python
# ❌ BAD: Guess the path
path = "docs/quickstart.md"

# ✅ GOOD: Look it up
path = memory.get_fact('locations', 'evolution_quickstart')
if path and memory.verify_path_exists(path):
    print(f"Quick start guide at: {path}")
```

## Benefits

### For AI Assistants
- ✅ Always have accurate information
- ✅ Confidence levels for all facts
- ✅ No more hallucinated file paths
- ✅ No more incorrect statistics
- ✅ Persistent memory across sessions

### For Users
- ✅ Trust the AI's responses
- ✅ Reproducible information
- ✅ Verifiable claims
- ✅ Consistent answers
- ✅ Up-to-date project state

## Maintenance

### Regular Updates

```bash
# Update memory with current state
python scripts/initialize_ai_memory.py
```

Run this:
- After major changes
- After reorganizations
- Before important operations
- Periodically (weekly)

### Verification

```bash
# Verify facts are still accurate
python scripts/verify_ai_facts.py
```

Run this:
- Before making claims
- After major changes
- When in doubt

### Backup

```python
# Backup memory
memory.export_memory('backups/memory_2024-10-26.json')

# Restore from backup
memory.import_memory('backups/memory_2024-10-26.json')
```

## Example Workflow

### 1. Initialize (Once)
```bash
python scripts/initialize_ai_memory.py
```

### 2. Use in AI Operations
```python
from core.ai_memory_system import get_memory

memory = get_memory()

# Before stating a fact
champion = memory.get_fact('tournament', 'champion_strategy')
# Use verified champion value

# Before referencing a file
path = memory.get_fact('locations', 'evolution_guide')
# Use verified path

# Before citing statistics
win_rate = memory.get_fact('tournament', 'champion_win_rate')
# Use verified statistic
```

### 3. Update After Changes
```bash
# After reorganizing files
python scripts/initialize_ai_memory.py

# Verify everything is correct
python scripts/verify_ai_facts.py
```

## Integration with Ollama

For Ollama AI integration:

1. **System Prompt Addition**:
```
You have access to an AI memory system at ai_memory.json that contains
verified facts about the project. Always check this memory before making
claims. Use get_memory() to access facts and verify_fact() to check claims.
Never state facts without verifying them first.
```

2. **Tool Registration**:
```python
# Register memory tools with Ollama
tools = [
    {
        "name": "get_fact",
        "description": "Get a verified fact from memory",
        "parameters": {
            "category": "string",
            "key": "string",
        }
    },
    {
        "name": "verify_fact",
        "description": "Verify a claim against memory",
        "parameters": {
            "category": "string",
            "key": "string",
            "value": "any",
        }
    },
]
```

3. **Pre-Check Hook**:
```python
def before_response(query, response):
    """Hook to verify facts before responding."""
    # Extract claims from response
    claims = extract_claims(response)
    
    # Verify each claim
    for claim in claims:
        if not verify_claim(claim):
            # Replace with verified fact or remove claim
            response = update_response(response, claim)
    
    return response
```

## Conclusion

The AI Memory System provides a robust foundation for preventing hallucinations by maintaining verified, persistent state. By always checking facts before stating them, AI assistants can provide accurate, trustworthy information.

**Key Principle**: *"Don't guess—verify!"*

