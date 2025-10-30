import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from drl.drl_training import train_dqn
from drl.drl_environment import MinesEnv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=2000)
    ap.add_argument('--cells', type=int, default=25)
    args = ap.parse_args()

    print(f"Starting DQN training with {args.episodes} episodes on {args.cells} cells...", flush=True)
    
    try:
        agent = train_dqn(episodes=args.episodes, n_cells=args.cells)
        print("Training completed successfully!", flush=True)
        
        # quick eval
        print("Running evaluation...", flush=True)
        env = MinesEnv(n_cells=args.cells)
        s = env.reset(); done = False; reward = 0.0
        steps = 0
        while not done and steps < 100:  # Prevent infinite loops
            valid = [False]*(env.n_cells+1)
            for a in env.valid_actions(): valid[a] = True
            a = agent.act(s, valid_mask=valid)
            r = env.step(a)
            s = r.next_state; done = r.done; reward += r.reward
            steps += 1
        print(f"Eval reward: {reward:.3f} (steps: {steps})", flush=True)
        
    except Exception as e:
        print(f"Error during training: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
