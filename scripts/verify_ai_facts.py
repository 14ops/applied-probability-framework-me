"""
Verify AI Facts - Check Statements Against Memory

Use this script to verify any statement or claim against
the stored memory system. Prevents hallucinations by checking
facts before stating them.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from core.ai_memory_system import get_memory
from typing import List, Dict, Any


class FactVerifier:
    """Verify statements against AI memory."""
    
    def __init__(self):
        self.memory = get_memory("ai_memory.json")
    
    def verify_project_claim(self, claim: str, expected_value: Any) -> Dict[str, Any]:
        """
        Verify a project-related claim.
        
        Returns:
            {
                'verified': bool,
                'stored_value': Any,
                'match': bool,
                'confidence': str,
            }
        """
        stored = self.memory.get_fact('project', claim)
        
        return {
            'verified': stored is not None,
            'stored_value': stored,
            'match': stored == expected_value,
            'confidence': 'high' if stored == expected_value else 'low',
        }
    
    def verify_file_exists(self, filepath: str) -> Dict[str, Any]:
        """Verify if a file exists in tracked structure."""
        exists = self.memory.verify_path_exists(filepath)
        location = self.memory.get_file_location(Path(filepath).name)
        
        return {
            'exists': exists,
            'tracked_location': location,
            'confidence': 'high' if exists else 'medium',
        }
    
    def verify_statistic(self, category: str, key: str, 
                        claimed_value: Any) -> Dict[str, Any]:
        """Verify a statistic."""
        stored = self.memory.get_fact(category, key)
        
        if stored is None:
            return {
                'verified': False,
                'reason': 'No stored value',
                'confidence': 'none',
            }
        
        # Handle numeric comparisons with tolerance
        if isinstance(stored, (int, float)) and isinstance(claimed_value, (int, float)):
            tolerance = 0.01  # 1% tolerance
            diff = abs(stored - claimed_value)
            relative_diff = diff / stored if stored != 0 else diff
            
            match = relative_diff <= tolerance
            
            return {
                'verified': True,
                'stored_value': stored,
                'claimed_value': claimed_value,
                'match': match,
                'difference': diff,
                'relative_difference': relative_diff,
                'confidence': 'high' if match else 'low',
            }
        
        # Exact match for non-numeric
        return {
            'verified': True,
            'stored_value': stored,
            'claimed_value': claimed_value,
            'match': stored == claimed_value,
            'confidence': 'high' if stored == claimed_value else 'low',
        }
    
    def get_verified_fact(self, category: str, key: str) -> Any:
        """Get a verified fact with confidence."""
        value = self.memory.get_fact(category, key)
        
        if value is None:
            return {
                'found': False,
                'value': None,
                'confidence': 'none',
            }
        
        return {
            'found': True,
            'value': value,
            'confidence': 'high',
        }
    
    def list_available_facts(self, category: str = None) -> List[str]:
        """List all available facts in a category."""
        if category:
            if category in self.memory.memory['facts']['verified']:
                return list(self.memory.memory['facts']['verified'][category].keys())
            return []
        
        # All categories
        facts = []
        for cat, items in self.memory.memory['facts']['verified'].items():
            for key in items.keys():
                facts.append(f"{cat}.{key}")
        return facts
    
    def check_statement(self, statement: str) -> Dict[str, Any]:
        """
        Check if a statement can be verified.
        
        This is a simple keyword-based checker.
        """
        statement_lower = statement.lower()
        
        results = {
            'statement': statement,
            'verifications': [],
            'overall_confidence': 'unknown',
        }
        
        # Check for common claims
        if 'win rate' in statement_lower:
            if 'hybrid' in statement_lower or 'champion' in statement_lower:
                result = self.get_verified_fact('tournament', 'champion_win_rate')
                results['verifications'].append({
                    'claim': 'champion win rate',
                    'result': result,
                })
        
        if 'champion' in statement_lower:
            result = self.get_verified_fact('tournament', 'champion_strategy')
            results['verifications'].append({
                'claim': 'champion name',
                'result': result,
            })
        
        if 'tournament' in statement_lower and 'million' in statement_lower:
            result = self.get_verified_fact('tournament', 'total_games_analyzed')
            results['verifications'].append({
                'claim': 'total games',
                'result': result,
            })
        
        # Determine overall confidence
        if results['verifications']:
            confidences = [v['result']['confidence'] for v in results['verifications']]
            if all(c == 'high' for c in confidences):
                results['overall_confidence'] = 'high'
            elif any(c == 'high' for c in confidences):
                results['overall_confidence'] = 'medium'
            else:
                results['overall_confidence'] = 'low'
        
        return results


def main():
    """Interactive fact verification."""
    print("="*80)
    print("🔍 AI Fact Verification System")
    print("="*80)
    
    verifier = FactVerifier()
    
    # Show available facts
    print("\n📚 Available fact categories:")
    facts = verifier.list_available_facts()
    categories = set(f.split('.')[0] for f in facts)
    for cat in sorted(categories):
        cat_facts = [f for f in facts if f.startswith(cat + '.')]
        print(f"  {cat}: {len(cat_facts)} facts")
    
    # Example verifications
    print("\n" + "="*80)
    print("🧪 Example Verifications")
    print("="*80)
    
    # Verify champion
    print("\n1. Verifying champion name...")
    result = verifier.get_verified_fact('tournament', 'champion_strategy')
    if result['found']:
        print(f"   ✓ Champion: {result['value']} (Confidence: {result['confidence']})")
    else:
        print("   ✗ Not found in memory")
    
    # Verify win rate
    print("\n2. Verifying champion win rate...")
    result = verifier.verify_statistic('tournament', 'champion_win_rate', 0.87)
    if result['verified'] and result['match']:
        print(f"   ✓ Win rate: {result['stored_value']}% (Confidence: {result['confidence']})")
    else:
        print(f"   ✗ Mismatch or not found")
    
    # Verify file location
    print("\n3. Verifying file location...")
    result = verifier.verify_file_exists('docs/evolution/QUICKSTART.md')
    if result['exists']:
        print(f"   ✓ File exists at: {result['tracked_location']}")
    else:
        print("   ✗ File not found in tracked structure")
    
    # Check full statement
    print("\n4. Checking statement...")
    statement = "Hybrid Ultimate is the champion with 0.87% win rate after 10 million games"
    result = verifier.check_statement(statement)
    print(f"   Statement: '{statement}'")
    print(f"   Verifications: {len(result['verifications'])}")
    print(f"   Overall Confidence: {result['overall_confidence']}")
    
    for v in result['verifications']:
        print(f"     - {v['claim']}: {v['result']['value']} ({v['result']['confidence']})")
    
    print("\n" + "="*80)
    print("✅ Verification Complete!")
    print("="*80)
    
    print("\n💡 Use this system to verify any claim before stating it!")
    print("   This prevents hallucinations by checking against stored facts.")


if __name__ == '__main__':
    main()

