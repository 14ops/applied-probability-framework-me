"""
Quick Tournament Test - 10K games to verify everything works before scaling to 10M
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

# Run the mega tournament with fewer games for testing
import examples.mega_tournament as tournament

if __name__ == '__main__':
    # Quick test with 10K games
    print("Running quick tournament test with 10,000 games...")
    stats = tournament.run_tournament(total_games=10_000, seed=42)
    tournament.print_final_rankings(stats)
    
    # Visualize
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    plot_file = results_dir / 'quick_tournament_test.png'
    
    try:
        tournament.visualize_results(stats, str(plot_file))
        print("\n✅ Test successful! Ready for 10 million games.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

