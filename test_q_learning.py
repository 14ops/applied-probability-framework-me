"""
Quick test to verify Q-Learning is working
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'python'))

from adaptive_takeshi import AdaptiveTakeshiStrategy
from game_simulator import GameSimulator

# Create Q-learning strategy
strategy = AdaptiveTakeshiStrategy()

print("=" * 60)
print("Q-Learning Test")
print("=" * 60)
print(f"Strategy: {strategy.name}")
print(f"Q-Learning enabled: {strategy.use_q_learning}")
print(f"Experience Replay enabled: {strategy.use_experience_replay}")
print(f"Q-Matrix initialized: {strategy.q_matrix is not None}")
print(f"Replay Buffer initialized: {strategy.replay_buffer is not None}")
print()

# Play one game
print("Playing one test game...")
game = GameSimulator(board_size=5, mine_count=3)
game.initialize_game()

moves = 0
done = False

while not done and moves < 5:
    # Get valid actions
    valid_actions = []
    for r in range(5):
        for c in range(5):
            if not game.revealed[r][c]:
                valid_actions.append((r, c))
    
    if not valid_actions:
        break
    
    # Get state
    state = {
        'revealed': game.revealed.tolist(),
        'board': game.board.tolist(),
        'board_size': game.board_size,
        'mine_count': game.mine_count,
        'clicks_made': moves
    }
    
    # Select action
    action = strategy.select_action(state, valid_actions)
    print(f"Move {moves + 1}: Click {action}")
    
    # Take action
    success, result = game.click_cell(action[0], action[1])
    moves += 1
    
    # Get next state
    next_state = {
        'revealed': game.revealed.tolist(),
        'board': game.board.tolist(),
        'board_size': game.board_size,
        'mine_count': game.mine_count,
        'clicks_made': moves
    }
    
    # Determine reward and done
    if result == "Mine":
        reward = -1.0
        done = True
        print(f"  💣 Hit a mine! Reward: {reward}")
    elif moves >= game.max_clicks:
        reward = 2.0
        done = True
        print(f"  🎉 Won! Reward: {reward}")
    else:
        reward = 0.1
        print(f"  ✅ Safe! Reward: {reward}")
    
    # Update strategy (this triggers Q-learning)
    strategy.update(state, action, reward, next_state, done)

print()
print("=" * 60)
print("After 1 game:")
print("=" * 60)
print(f"Q-Matrix size: {len(strategy.q_matrix.q_table_a)} states")
print(f"Total Q-updates: {strategy.q_matrix.updates}")
print(f"Epsilon: {strategy.q_matrix.epsilon:.4f}")
print(f"Replay buffer size: {len(strategy.replay_buffer)}")
print()

if len(strategy.q_matrix.q_table_a) > 0:
    print("✅ Q-Learning is working!")
    print("   States are being learned and stored in the Q-matrix")
else:
    print("⚠️  Q-Matrix is still empty - may need more games")

print()
print("Now run the GUI and play a few games to see the matrices fill up!")
print("  python src/python/gui_game.py")

