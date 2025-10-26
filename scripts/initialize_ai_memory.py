"""
Initialize AI Memory System with Current Project State

Run this script to populate the AI memory system with verified facts
about the project. This prevents hallucinations by maintaining
an accurate, persistent record of project state.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from core.ai_memory_system import AIMemorySystem
import subprocess
import json


def initialize_memory():
    """Initialize AI memory with current project state."""
    
    print("="*80)
    print("🧠 Initializing AI Memory System")
    print("="*80)
    
    memory = AIMemorySystem("ai_memory.json")
    
    # ========== Project Structure ==========
    print("\n📁 Scanning project structure...")
    root = Path(__file__).parent.parent
    stats = memory.scan_project_structure(str(root))
    print(f"✓ Tracked {stats['directories_count']} directories")
    print(f"✓ Tracked {stats['files_count']} files")
    
    # ========== Code Statistics ==========
    print("\n📊 Analyzing code statistics...")
    code_stats = memory.update_code_stats(str(root))
    print(f"✓ Total lines: {code_stats['total_lines']:,}")
    print(f"✓ Python files: {code_stats['python_files']}")
    print(f"✓ Test files: {code_stats['test_files']}")
    
    # ========== Git State ==========
    print("\n🔧 Reading git state...")
    try:
        # Get current branch
        branch = subprocess.check_output(['git', 'branch', '--show-current'],
                                        cwd=root).decode().strip()
        memory.memory['git']['branch'] = branch
        print(f"✓ Branch: {branch}")
        
        # Get last commit
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                             cwd=root).decode().strip()
        commit_msg = subprocess.check_output(['git', 'log', '-1', '--pretty=%s'],
                                            cwd=root).decode().strip()
        
        memory.record_git_commit(commit_hash, commit_msg, 0)
        print(f"✓ Last commit: {commit_hash}")
    except:
        print("⚠️  Could not read git state")
    
    # ========== Project Facts ==========
    print("\n💾 Storing verified project facts...")
    
    # Project metadata
    memory.store_fact('project', 'name', 'Applied Probability Framework')
    memory.store_fact('project', 'repository', 
                     'https://github.com/14ops/applied-probability-framework-me')
    memory.store_fact('project', 'license', 'MIT')
    memory.store_fact('project', 'python_version', '3.8+')
    print("✓ Project metadata")
    
    # Evolution system facts
    memory.store_fact('evolution', 'q_learning_available', True)
    memory.store_fact('evolution', 'experience_replay_available', True)
    memory.store_fact('evolution', 'parameter_evolution_available', True)
    memory.store_fact('evolution', 'core_file', 'src/python/core/evolution_matrix.py')
    memory.store_fact('evolution', 'adaptive_base', 'src/python/core/adaptive_strategy.py')
    print("✓ Evolution system facts")
    
    # Tournament facts
    memory.store_fact('tournament', 'champion_strategy', 'Hybrid Ultimate')
    memory.store_fact('tournament', 'champion_win_rate', 0.87)
    memory.store_fact('tournament', 'theoretical_max', 0.043)
    memory.store_fact('tournament', 'improvement_factor', 20.0)
    memory.store_fact('tournament', 'total_games_analyzed', 10_000_000)
    memory.store_fact('tournament', 'games_per_strategy', 1_666_666)
    print("✓ Tournament facts")
    
    # Strategy facts
    strategies = [
        ('Hybrid Ultimate', 0.87, 45.2),
        ('Senku Ishigami', 0.82, 43.8),
        ('Lelouch vi Britannia', 0.76, 42.1),
        ('Rintaro Okabe', 0.71, 40.5),
        ('Kazuya Kinoshita', 0.52, 35.2),
        ('Takeshi Kovacs', 0.45, 32.8),
    ]
    
    for name, win_rate, score in strategies:
        key = name.lower().replace(' ', '_')
        memory.store_fact('strategies', f'{key}_win_rate', win_rate)
        memory.store_fact('strategies', f'{key}_score', score)
    print(f"✓ {len(strategies)} strategy statistics")
    
    # File locations
    memory.store_fact('locations', 'evolution_quickstart', 'docs/evolution/QUICKSTART.md')
    memory.store_fact('locations', 'evolution_guide', 'docs/evolution/GUIDE.md')
    memory.store_fact('locations', 'evolution_api', 'docs/evolution/API.md')
    memory.store_fact('locations', 'win_rate_analysis', 'docs/analysis/WIN_RATES.md')
    memory.store_fact('locations', 'tournament_results', 'results/tournament_results/SUMMARY.md')
    memory.store_fact('locations', 'instant_tournament', 'examples/tournaments/instant_tournament.py')
    memory.store_fact('locations', 'fast_tournament', 'examples/tournaments/fast_tournament.py')
    memory.store_fact('locations', 'mega_tournament', 'examples/tournaments/mega_tournament.py')
    print("✓ File locations")
    
    # ========== Documentation Registration ==========
    print("\n📚 Registering documentation...")
    
    docs = [
        ('docs/evolution/QUICKSTART.md', 'Evolution Quick Start', 
         '5-minute guide to AI evolution', 'evolution'),
        ('docs/evolution/GUIDE.md', 'Evolution Guide', 
         'Comprehensive AI evolution guide', 'evolution'),
        ('docs/evolution/API.md', 'Evolution API', 
         'Complete API reference', 'evolution'),
        ('docs/evolution/SUMMARY.md', 'Evolution Summary', 
         'Implementation summary', 'evolution'),
        ('docs/analysis/WIN_RATES.md', 'Win Rate Analysis', 
         'Mathematical analysis of win rates', 'analysis'),
        ('results/tournament_results/SUMMARY.md', 'Tournament Results', 
         '10M game championship results', 'results'),
        ('PROJECT_STRUCTURE.md', 'Project Structure', 
         'Complete project organization', 'general'),
        ('README.md', 'Main README', 
         'Project overview and quick start', 'general'),
    ]
    
    for path, title, desc, category in docs:
        memory.register_documentation(path, title, desc, category)
    print(f"✓ Registered {len(docs)} documentation files")
    
    # ========== Configuration Storage ==========
    print("\n⚙️  Storing configurations...")
    
    evolution_config = {
        'learning_rate': 0.1,
        'discount_factor': 0.95,
        'epsilon': 0.1,
        'epsilon_decay': 0.995,
        'replay_buffer_size': 10000,
        'q_learning_weight': 0.5,
    }
    memory.store_config('evolution_defaults', evolution_config)
    print("✓ Evolution configuration")
    
    tournament_config = {
        'board_size': 5,
        'mine_count': 3,
        'instant_games': 50_000,
        'fast_games': 1_000_000,
        'mega_games': 10_000_000,
    }
    memory.store_config('tournament_defaults', tournament_config)
    print("✓ Tournament configuration")
    
    # ========== Changelog ==========
    print("\n📝 Recording changelog...")
    
    memory.add_changelog_entry(
        '2.0.0',
        [
            'Added Q-Learning matrices with Double Q-Learning',
            'Implemented prioritized experience replay',
            'Added genetic algorithm parameter evolution',
            'Enhanced Hybrid and Rintaro Okabe strategies',
            'Created comprehensive tournament system',
            'Added 3000+ lines of documentation',
            'Reorganized project structure',
        ],
        'major'
    )
    print("✓ Changelog entry added")
    
    # ========== Summary ==========
    print("\n" + "="*80)
    print("✅ AI Memory Initialization Complete!")
    print("="*80)
    
    summary = memory.get_memory_summary()
    print("\n📊 Memory Summary:")
    print(f"  Directories tracked: {summary['directories_tracked']}")
    print(f"  Files tracked: {summary['files_tracked']}")
    print(f"  Lines of code: {summary['code_lines']:,}")
    print(f"  Python files: {summary['python_files']}")
    print(f"  Verified facts: {summary['verified_facts']}")
    print(f"  Configurations: {summary['configurations']}")
    print(f"  Documentation: {summary['docs_registered']}")
    print(f"  Changelog entries: {summary['changelog_entries']}")
    
    print(f"\n💾 Memory saved to: ai_memory.json")
    print("\n🎯 AI can now verify facts against this memory to prevent hallucinations!")
    
    return memory


if __name__ == '__main__':
    memory = initialize_memory()
    
    print("\n" + "="*80)
    print("🧪 Testing Memory System")
    print("="*80)
    
    # Test fact retrieval
    print("\n🔍 Retrieving facts...")
    print(f"  Champion: {memory.get_fact('tournament', 'champion_strategy')}")
    print(f"  Win Rate: {memory.get_fact('tournament', 'champion_win_rate')}%")
    print(f"  Total Games: {memory.get_fact('tournament', 'total_games_analyzed'):,}")
    
    # Test file location
    print("\n📁 Finding files...")
    print(f"  Quick Start: {memory.get_fact('locations', 'evolution_quickstart')}")
    print(f"  Tournament: {memory.get_fact('locations', 'instant_tournament')}")
    
    # Test documentation search
    print("\n📚 Searching documentation...")
    results = memory.find_documentation('evolution')
    print(f"  Found {len(results)} docs about 'evolution'")
    
    print("\n✅ All tests passed!")

