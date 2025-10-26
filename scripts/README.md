# Scripts

Utility scripts for project maintenance and AI memory management.

## AI Memory System Scripts

### 🧠 **initialize_ai_memory.py**
Initialize the AI memory system with current project state.

**Purpose**: Creates `ai_memory.json` containing verified facts about the project to prevent AI hallucinations.

**Usage**:
```bash
python scripts/initialize_ai_memory.py
```

**Creates**:
- Complete file structure index
- Code statistics (lines, files)
- Git commit history
- Verified project facts
- Documentation registry
- Configuration storage

**Run after**:
- Project reorganization
- Major changes
- Before important operations
- Periodically (weekly)

---

### 🔍 **verify_ai_facts.py**
Verify facts interactively against stored memory.

**Purpose**: Check any claim or statement against verified project data.

**Usage**:
```bash
python scripts/verify_ai_facts.py
```

**Features**:
- List all available facts
- Verify specific claims
- Check file locations
- Validate statistics
- Interactive verification

---

## How It Works

### Preventing Hallucinations

The AI Memory System maintains a persistent JSON file (`ai_memory.json`) that stores:

1. **Project Structure** - All files and directories
2. **Code Statistics** - Lines of code, file counts
3. **Verified Facts** - Tournament results, configurations
4. **Git History** - Commits and changes
5. **Documentation** - Searchable index

### Usage Flow

```
1. Initialize Memory
   python scripts/initialize_ai_memory.py
   
2. AI Checks Facts
   - Before making claims
   - Before stating statistics
   - Before referencing files
   
3. Verify if Needed
   python scripts/verify_ai_facts.py
   
4. Update After Changes
   python scripts/initialize_ai_memory.py
```

---

## Example Verifications

### Check Champion
```python
from core.ai_memory_system import get_memory

memory = get_memory()
champion = memory.get_fact('tournament', 'champion_strategy')
# Returns: 'Hybrid Ultimate' (verified!)
```

### Check File Location
```python
path = memory.get_fact('locations', 'evolution_quickstart')
exists = memory.verify_path_exists(path)
# Returns: True if file exists at that path
```

### Check Statistic
```python
win_rate = memory.get_fact('tournament', 'champion_win_rate')
# Returns: 0.87 (verified from tournament results!)
```

---

## Current Memory State

After initialization, the memory contains:

- **77 directories** tracked
- **1,187 files** indexed
- **16,230 lines** of code
- **87 Python files**
- **35 verified facts**
- **2 configurations**
- **8 documentation files**

---

## Benefits

### For AI
✅ Always accurate information
✅ No hallucinated facts
✅ Persistent memory
✅ Verifiable claims

### For Users
✅ Trust AI responses
✅ Reproducible information
✅ Consistent answers
✅ Up-to-date project state

---

## Documentation

See [AI_MEMORY_SYSTEM.md](../docs/AI_MEMORY_SYSTEM.md) for complete documentation.

---

**Key Principle**: *"Don't guess—verify!"*

