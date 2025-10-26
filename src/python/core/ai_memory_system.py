"""
AI Memory System - Prevents Hallucinations Through Persistent State Tracking

This system maintains accurate records of:
- Project structure and file locations
- Code statistics and metrics
- Git state and commit history
- Configuration values
- Test results and coverage
- Documentation locations
- Known facts and verified information

Prevents hallucinations by always checking against stored state.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib


class AIMemorySystem:
    """
    Persistent memory system for AI assistants.
    
    Stores verified facts, project state, and historical data
    to prevent hallucinations and maintain consistency.
    """
    
    def __init__(self, memory_file: str = "ai_memory.json"):
        """
        Initialize memory system.
        
        Args:
            memory_file: Path to persistent memory storage
        """
        self.memory_file = Path(memory_file)
        self.memory: Dict[str, Any] = self._load_memory()
        
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from disk or create new."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load memory: {e}")
                return self._create_empty_memory()
        return self._create_empty_memory()
    
    def _create_empty_memory(self) -> Dict[str, Any]:
        """Create empty memory structure."""
        return {
            'project': {
                'name': 'Applied Probability Framework',
                'version': '1.0.0',
                'repository': 'https://github.com/14ops/applied-probability-framework-me',
            },
            'structure': {
                'directories': {},
                'files': {},
                'last_scanned': None,
            },
            'code_stats': {
                'total_lines': 0,
                'python_files': 0,
                'test_files': 0,
                'last_updated': None,
            },
            'git': {
                'branch': 'main',
                'last_commit': None,
                'commits': [],
            },
            'facts': {
                'verified': {},
                'deprecated': {},
            },
            'configurations': {},
            'test_results': [],
            'documentation': {},
            'changelog': [],
            'metadata': {
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'updates_count': 0,
            }
        }
    
    def save(self):
        """Save memory to disk."""
        self.memory['metadata']['last_updated'] = datetime.now().isoformat()
        self.memory['metadata']['updates_count'] += 1
        
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    # ========== Project Structure ==========
    
    def scan_project_structure(self, root_dir: str = "."):
        """Scan and store project structure."""
        root = Path(root_dir)
        
        directories = {}
        files = {}
        
        for path in root.rglob("*"):
            rel_path = str(path.relative_to(root))
            
            # Skip hidden and cache directories
            if any(part.startswith('.') for part in path.parts):
                continue
            if '__pycache__' in path.parts or 'node_modules' in path.parts:
                continue
            
            if path.is_dir():
                directories[rel_path] = {
                    'exists': True,
                    'type': 'directory',
                    'last_verified': datetime.now().isoformat(),
                }
            elif path.is_file():
                files[rel_path] = {
                    'exists': True,
                    'size': path.stat().st_size,
                    'type': path.suffix,
                    'last_verified': datetime.now().isoformat(),
                }
        
        self.memory['structure']['directories'] = directories
        self.memory['structure']['files'] = files
        self.memory['structure']['last_scanned'] = datetime.now().isoformat()
        
        self.save()
        
        return {
            'directories_count': len(directories),
            'files_count': len(files),
        }
    
    def verify_path_exists(self, path: str) -> bool:
        """Verify if a path exists (check against stored state)."""
        return (path in self.memory['structure']['files'] or 
                path in self.memory['structure']['directories'])
    
    def get_file_location(self, filename: str) -> Optional[str]:
        """Find file location in project."""
        for path in self.memory['structure']['files']:
            if path.endswith(filename):
                return path
        return None
    
    # ========== Code Statistics ==========
    
    def update_code_stats(self, root_dir: str = "."):
        """Update code statistics."""
        root = Path(root_dir)
        
        total_lines = 0
        python_files = 0
        test_files = 0
        
        for py_file in root.rglob("*.py"):
            if any(part.startswith('.') for part in py_file.parts):
                continue
            if '__pycache__' in py_file.parts:
                continue
            
            python_files += 1
            
            if 'test' in py_file.name.lower():
                test_files += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except:
                pass
        
        self.memory['code_stats'] = {
            'total_lines': total_lines,
            'python_files': python_files,
            'test_files': test_files,
            'last_updated': datetime.now().isoformat(),
        }
        
        self.save()
        
        return self.memory['code_stats']
    
    def get_code_stats(self) -> Dict[str, Any]:
        """Get current code statistics."""
        return self.memory['code_stats']
    
    # ========== Git Tracking ==========
    
    def record_git_commit(self, commit_hash: str, message: str, files_changed: int):
        """Record a git commit."""
        commit = {
            'hash': commit_hash,
            'message': message,
            'files_changed': files_changed,
            'timestamp': datetime.now().isoformat(),
        }
        
        self.memory['git']['commits'].append(commit)
        self.memory['git']['last_commit'] = commit
        
        # Keep only last 100 commits
        if len(self.memory['git']['commits']) > 100:
            self.memory['git']['commits'] = self.memory['git']['commits'][-100:]
        
        self.save()
    
    def get_last_commit(self) -> Optional[Dict[str, Any]]:
        """Get last recorded commit."""
        return self.memory['git'].get('last_commit')
    
    def get_commit_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent commit history."""
        return self.memory['git']['commits'][-limit:]
    
    # ========== Verified Facts ==========
    
    def store_fact(self, category: str, key: str, value: Any, 
                   verification_method: str = "manual"):
        """
        Store a verified fact.
        
        Args:
            category: Category of fact (e.g., 'file_locations', 'statistics')
            key: Unique key for the fact
            value: The fact value
            verification_method: How it was verified
        """
        if category not in self.memory['facts']['verified']:
            self.memory['facts']['verified'][category] = {}
        
        self.memory['facts']['verified'][category][key] = {
            'value': value,
            'verification_method': verification_method,
            'timestamp': datetime.now().isoformat(),
            'hash': hashlib.md5(str(value).encode()).hexdigest(),
        }
        
        self.save()
    
    def get_fact(self, category: str, key: str) -> Optional[Any]:
        """Retrieve a verified fact."""
        if category in self.memory['facts']['verified']:
            if key in self.memory['facts']['verified'][category]:
                return self.memory['facts']['verified'][category][key]['value']
        return None
    
    def verify_fact(self, category: str, key: str, expected_value: Any) -> bool:
        """Verify if a fact matches stored value."""
        stored = self.get_fact(category, key)
        return stored == expected_value
    
    def deprecate_fact(self, category: str, key: str, reason: str):
        """Mark a fact as deprecated."""
        if category in self.memory['facts']['verified']:
            if key in self.memory['facts']['verified'][category]:
                fact = self.memory['facts']['verified'][category].pop(key)
                
                if category not in self.memory['facts']['deprecated']:
                    self.memory['facts']['deprecated'][category] = {}
                
                fact['deprecation_reason'] = reason
                fact['deprecation_time'] = datetime.now().isoformat()
                
                self.memory['facts']['deprecated'][category][key] = fact
                
                self.save()
    
    # ========== Configuration Tracking ==========
    
    def store_config(self, name: str, config: Dict[str, Any]):
        """Store a configuration."""
        self.memory['configurations'][name] = {
            'config': config,
            'timestamp': datetime.now().isoformat(),
        }
        self.save()
    
    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a configuration."""
        if name in self.memory['configurations']:
            return self.memory['configurations'][name]['config']
        return None
    
    # ========== Test Results ==========
    
    def record_test_run(self, passed: int, failed: int, total: int, 
                       coverage: float, duration: float):
        """Record test run results."""
        result = {
            'passed': passed,
            'failed': failed,
            'total': total,
            'coverage': coverage,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
        }
        
        self.memory['test_results'].append(result)
        
        # Keep only last 50 test runs
        if len(self.memory['test_results']) > 50:
            self.memory['test_results'] = self.memory['test_results'][-50:]
        
        self.save()
    
    def get_latest_test_results(self) -> Optional[Dict[str, Any]]:
        """Get most recent test results."""
        if self.memory['test_results']:
            return self.memory['test_results'][-1]
        return None
    
    # ========== Documentation Tracking ==========
    
    def register_documentation(self, doc_path: str, title: str, 
                              description: str, category: str):
        """Register documentation file."""
        self.memory['documentation'][doc_path] = {
            'title': title,
            'description': description,
            'category': category,
            'timestamp': datetime.now().isoformat(),
        }
        self.save()
    
    def find_documentation(self, search_term: str) -> List[Dict[str, Any]]:
        """Find documentation by search term."""
        results = []
        
        for path, info in self.memory['documentation'].items():
            if (search_term.lower() in path.lower() or
                search_term.lower() in info['title'].lower() or
                search_term.lower() in info['description'].lower()):
                results.append({
                    'path': path,
                    **info
                })
        
        return results
    
    # ========== Changelog ==========
    
    def add_changelog_entry(self, version: str, changes: List[str], 
                           change_type: str = "feature"):
        """Add changelog entry."""
        entry = {
            'version': version,
            'changes': changes,
            'type': change_type,
            'timestamp': datetime.now().isoformat(),
        }
        
        self.memory['changelog'].append(entry)
        self.save()
    
    def get_changelog(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent changelog entries."""
        return self.memory['changelog'][-limit:]
    
    # ========== Utility Methods ==========
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of stored memory."""
        return {
            'directories_tracked': len(self.memory['structure']['directories']),
            'files_tracked': len(self.memory['structure']['files']),
            'code_lines': self.memory['code_stats']['total_lines'],
            'python_files': self.memory['code_stats']['python_files'],
            'commits_recorded': len(self.memory['git']['commits']),
            'verified_facts': sum(len(facts) for facts in self.memory['facts']['verified'].values()),
            'configurations': len(self.memory['configurations']),
            'test_runs': len(self.memory['test_results']),
            'docs_registered': len(self.memory['documentation']),
            'changelog_entries': len(self.memory['changelog']),
            'last_updated': self.memory['metadata']['last_updated'],
            'updates_count': self.memory['metadata']['updates_count'],
        }
    
    def export_memory(self, filepath: str):
        """Export memory to file."""
        with open(filepath, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def import_memory(self, filepath: str):
        """Import memory from file."""
        with open(filepath, 'r') as f:
            self.memory = json.load(f)
        self.save()
    
    def clear_memory(self, confirm: bool = False):
        """Clear all memory (requires confirmation)."""
        if confirm:
            self.memory = self._create_empty_memory()
            self.save()


# ========== Convenience Functions ==========

_global_memory: Optional[AIMemorySystem] = None


def get_memory(memory_file: str = "ai_memory.json") -> AIMemorySystem:
    """Get global memory system instance."""
    global _global_memory
    if _global_memory is None:
        _global_memory = AIMemorySystem(memory_file)
    return _global_memory


def quick_scan(root_dir: str = ".") -> Dict[str, Any]:
    """Quick scan of project and update memory."""
    memory = get_memory()
    
    print("📊 Scanning project...")
    structure_stats = memory.scan_project_structure(root_dir)
    code_stats = memory.update_code_stats(root_dir)
    
    print(f"✓ Tracked {structure_stats['directories_count']} directories")
    print(f"✓ Tracked {structure_stats['files_count']} files")
    print(f"✓ Found {code_stats['python_files']} Python files")
    print(f"✓ Total {code_stats['total_lines']:,} lines of code")
    
    return memory.get_memory_summary()


def store_project_fact(key: str, value: Any):
    """Store a verified project fact."""
    memory = get_memory()
    memory.store_fact('project', key, value, 'verified')
    print(f"✓ Stored fact: {key} = {value}")


def verify_project_fact(key: str, value: Any) -> bool:
    """Verify a project fact against memory."""
    memory = get_memory()
    stored = memory.get_fact('project', key)
    
    if stored is None:
        print(f"⚠️  No stored value for: {key}")
        return False
    
    if stored == value:
        print(f"✓ Verified: {key} = {value}")
        return True
    else:
        print(f"✗ Mismatch: {key}")
        print(f"  Expected: {value}")
        print(f"  Stored: {stored}")
        return False


if __name__ == "__main__":
    # Example usage
    print("AI Memory System - Preventing Hallucinations\n")
    
    # Quick scan
    stats = quick_scan()
    
    print("\n📋 Memory Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Store some facts
    print("\n💾 Storing verified facts...")
    store_project_fact("champion_strategy", "Hybrid Ultimate")
    store_project_fact("champion_win_rate", 0.87)
    store_project_fact("total_tournament_games", 10_000_000)
    
    # Verify facts
    print("\n🔍 Verifying facts...")
    verify_project_fact("champion_strategy", "Hybrid Ultimate")
    verify_project_fact("champion_win_rate", 0.87)
    
    print("\n✅ Memory system active!")

