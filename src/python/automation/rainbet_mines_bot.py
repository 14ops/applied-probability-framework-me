"""
RainBet Mines Automation Bot
Advanced computer vision and automation for autonomous gameplay

Features:
- Template matching for board detection
- Anti-detection with human-like behavior
- Seamless OCR-free operation
- Integration with character strategies
- Real-time decision making
"""

import cv2
import numpy as np
import pyautogui
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import json
import logging

# Import our character strategies
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'python'))

from game.math import win_probability, payout_table_25_2
from character_strategies.takeshi_kovacs import TakeshiKovacsStrategy
from character_strategies.kazuya_kinoshita import KazuyaKinoshitaStrategy
from character_strategies.lelouch_vi_britannia import LelouchViBritanniaStrategy

# Anti-detection tools
sys.path.insert(0, str(Path(__file__).parent.parent))
from anti_detection_simulator import HumanBehaviorEmulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RainBetMinesBot:
    """
    Autonomous Mines bot for RainBet with computer vision and AI strategies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the bot with configuration.
        
        Args:
            config: Configuration dictionary with settings for:
                - character: Strategy to use
                - bet_amount: Initial bet
                - total_rounds: Maximum rounds to play
                - anti_detection: Enable human-like behavior
                - screenshot_region: Region of screen to capture
        """
        self.config = config or {}
        self.character_name = self.config.get('character', 'hybrid')
        self.bet_amount = self.config.get('bet_amount', 10.0)
        self.total_rounds = self.config.get('total_rounds', 100)
        self.anti_detection = self.config.get('anti_detection', True)
        
        # Initialize strategy
        self.strategy = self._create_strategy(self.character_name)
        
        # Tracking
        self.current_round = 0
        self.current_profit = 0.0
        self.bankroll = self.config.get('initial_bankroll', 1000.0)
        self.clicks_made = 0
        self.wins = 0
        self.losses = 0
        
        # Game state
        self.board_state = np.zeros((5, 5), dtype=bool)  # Revealed cells
        self.current_multiplier = 1.0
        self.round_active = False
        
        # Anti-detection
        self.human_emulator = HumanBehaviorEmulator(
            min_delay_ms=200,
            max_delay_ms=800,
            long_pause_interval=10,
            long_pause_min_s=5,
            long_pause_max_s=15
        ) if self.anti_detection else None
        
        # Screenshot region (to be configured)
        self.screenshot_region = self.config.get('screenshot_region', None)
        
        # Logging
        self.log_dir = Path('logs/automation')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_log = []
        
        logger.info(f"Bot initialized with character: {self.character_name}")
    
    def _create_strategy(self, character_name: str):
        """Create a strategy instance based on character name."""
        strategy_config = {
            'base_bet': self.bet_amount,
            'initial_bankroll': self.config.get('initial_bankroll', 1000.0)
        }
        
        if character_name.lower() == 'takeshi':
            return TakeshiKovacsStrategy(config=strategy_config)
        elif character_name.lower() == 'kazuya':
            return KazuyaKinoshitaStrategy(config=strategy_config)
        elif character_name.lower() == 'lelouch':
            return LelouchViBritanniaStrategy(config=strategy_config)
        else:
            logger.warning(f"Unknown character: {character_name}, using Hybrid")
            # Return a hybrid or default strategy
            return TakeshiKovacsStrategy(config=strategy_config)
    
    def calibrate(self):
        """
        Calibrate the bot by detecting the game window and tile positions.
        This should be run once before starting automation.
        """
        logger.info("Starting calibration...")
        
        # Take a screenshot of the game area
        screenshot = self._take_screenshot()
        if screenshot is None:
            logger.error("Failed to capture screenshot")
            return False
        
        # Detect tile grid
        tile_positions = self._detect_tile_grid(screenshot)
        if not tile_positions or len(tile_positions) != 25:
            logger.error(f"Failed to detect 25 tiles, found {len(tile_positions) if tile_positions else 0}")
            return False
        
        self.tile_positions = tile_positions
        logger.info(f"Calibration successful: detected {len(tile_positions)} tiles")
        
        # Detect cash-out button
        self.cashout_button_pos = self._detect_cashout_button(screenshot)
        if self.cashout_button_pos:
            logger.info(f"Cash-out button detected at {self.cashout_button_pos}")
        else:
            logger.warning("Cash-out button not detected, will use stored position")
            # Default position (needs to be customized)
            self.cashout_button_pos = (800, 600)
        
        return True
    
    def _take_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot of the game region."""
        try:
            if self.screenshot_region:
                x, y, w, h = self.screenshot_region
                screenshot = pyautogui.screenshot(region=(x, y, w, h))
            else:
                screenshot = pyautogui.screenshot()
            
            # Convert to OpenCV format
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return None
    
    def _detect_tile_grid(self, img: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Detect the 5x5 tile grid using template matching.
        
        Returns:
            List of (x, y) center positions for each tile
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to find a single tile template (if you have one)
        # For now, use a simple approach: detect rectangular regions
        
        # Method 1: Edge detection + contour finding
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for square-like regions
        tiles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 2000:  # Reasonable tile size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.8 < aspect_ratio < 1.2:  # Roughly square
                    center_x = x + w // 2
                    center_y = y + h // 2
                    tiles.append((center_x, center_y))
        
        # If we found tiles, organize them into a grid
        if len(tiles) >= 20:
            # Sort by position and organize into 5x5
            tiles_sorted = sorted(tiles, key=lambda t: (t[1], t[0]))
            
            # Group by row (approximate)
            grid = []
            current_row = [tiles_sorted[0]]
            for tile in tiles_sorted[1:]:
                if abs(tile[1] - current_row[0][1]) < 20:  # Same row
                    current_row.append(tile)
                else:
                    current_row.sort()
                    grid.append(current_row)
                    current_row = [tile]
            if current_row:
                current_row.sort()
                grid.append(current_row)
            
            # Flatten
            flattened = []
            for row in grid:
                flattened.extend(row)
            
            if len(flattened) >= 25:
                return flattened[:25]
        
        # Fallback: Return None and use manual calibration
        logger.warning("Grid detection failed, will need manual calibration")
        return None
    
    def _detect_cashout_button(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect the cash-out button position."""
        # Simple: look for a green button or specific color
        # In practice, you'd use template matching
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Look for green color (typical for cash-out buttons)
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find contours of green regions
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:  # Button-sized region
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                return (center_x, center_y)
        
        return None
    
    def _detect_board_state(self) -> Optional[np.ndarray]:
        """
        Detect current board state by checking each tile.
        
        Returns:
            5x5 numpy array where True = revealed, False = hidden
        """
        screenshot = self._take_screenshot()
        if screenshot is None:
            return None
        
        board_state = np.zeros((5, 5), dtype=bool)
        
        # Check each tile
        for i, (x, y) in enumerate(self.tile_positions):
            row = i // 5
            col = i % 5
            
            # Extract tile region
            tile_size = 30  # Approximate tile size
            x1 = max(0, x - tile_size)
            y1 = max(0, y - tile_size)
            x2 = min(screenshot.shape[1], x + tile_size)
            y2 = min(screenshot.shape[0], y + tile_size)
            
            tile_roi = screenshot[y1:y2, x1:x2]
            
            if tile_roi.size > 0:
                # Check if tile is revealed (color-based detection)
                # Unrevealed tiles are usually darker/gray
                # Revealed tiles might be green (safe) or red (mine)
                
                mean_color = np.mean(tile_roi, axis=(0, 1))
                
                # Very simple heuristic: bright colors = revealed
                if np.mean(mean_color) > 100:
                    board_state[row, col] = True
        
        return board_state
    
    def _click_tile(self, row: int, col: int):
        """Click a tile at the given position."""
        if not hasattr(self, 'tile_positions'):
            logger.error("Bot not calibrated")
            return False
        
        # Get tile position
        tile_idx = row * 5 + col
        if tile_idx >= len(self.tile_positions):
            logger.error(f"Invalid tile index: {tile_idx}")
            return False
        
        x, y = self.tile_positions[tile_idx]
        
        # Add offset if screenshot region is specified
        if self.screenshot_region:
            x += self.screenshot_region[0]
            y += self.screenshot_region[1]
        
        # Human-like mouse movement
        if self.human_emulator:
            self.human_emulator.apply_human_jitter("Click")
            # Move with slight curve
            self.human_emulator.simulate_mouse_drift(
                pyautogui.position(), 
                (x, y)
            )
        
        # Click
        pyautogui.click(x, y)
        
        # Post-click delay
        delay = random.uniform(0.3, 0.7)
        time.sleep(delay)
        
        logger.debug(f"Clicked tile ({row}, {col}) at ({x}, {y})")
        return True
    
    def _click_cashout(self):
        """Click the cash-out button."""
        if not hasattr(self, 'cashout_button_pos'):
            logger.error("Cash-out button position not known")
            return False
        
        x, y = self.cashout_button_pos
        if self.screenshot_region:
            x += self.screenshot_region[0]
            y += self.screenshot_region[1]
        
        # Human-like movement
        if self.human_emulator:
            self.human_emulator.apply_human_jitter("Click")
            self.human_emulator.simulate_mouse_drift(
                pyautogui.position(), 
                (x, y)
            )
        
        pyautogui.click(x, y)
        time.sleep(0.5)
        
        logger.info("Clicked cash-out button")
        return True
    
    def _select_next_tile(self) -> Optional[Tuple[int, int]]:
        """Use strategy to select the next tile to click."""
        # Get valid actions (unrevealed tiles)
        valid_actions = []
        for row in range(5):
            for col in range(5):
                if not self.board_state[row, col]:
                    valid_actions.append((row, col))
        
        if not valid_actions:
            return None
        
        # Get state for strategy
        state = {
            'board_size': 5,
            'mine_count': 3,  # Assume 3 mines (or make configurable)
            'clicks_made': self.clicks_made,
            'current_payout': self.current_multiplier,
            'revealed': self.board_state.tolist()
        }
        
        # Ask strategy for action
        try:
            action = self.strategy.select_action(state, valid_actions)
            return action
        except Exception as e:
            logger.error(f"Strategy error: {e}")
            # Fallback: random tile
            return random.choice(valid_actions)
    
    def _should_cash_out(self) -> bool:
        """Determine if we should cash out based on strategy."""
        # Get state
        valid_actions = []
        for row in range(5):
            for col in range(5):
                if not self.board_state[row, col]:
                    valid_actions.append((row, col))
        
        state = {
            'board_size': 5,
            'mine_count': 3,
            'clicks_made': self.clicks_made,
            'current_payout': self.current_multiplier,
            'revealed': self.board_state.tolist()
        }
        
        # Check if strategy returns None (cash-out signal)
        try:
            action = self.strategy.select_action(state, valid_actions)
            if action is None:
                return True
        except Exception:
            pass
        
        return False
    
    def run_round(self) -> Dict[str, Any]:
        """
        Execute a single round of Mines.
        
        Returns:
            Dictionary with round results
        """
        self.current_round += 1
        self.clicks_made = 0
        self.current_multiplier = 1.0
        self.round_active = True
        
        logger.info(f"Starting round {self.current_round}/{self.total_rounds}")
        
        # Detect initial board state
        self.board_state = self._detect_board_state()
        if self.board_state is None:
            logger.error("Failed to detect board state")
            return {'success': False, 'error': 'board_detection_failed'}
        
        # Decision loop
        max_clicks = 23  # 25 - 2 mines (assuming 2 mines)
        round_result = None
        
        for click in range(max_clicks):
            # Check if should cash out
            if self._should_cash_out():
                logger.info(f"Strategy decided to cash out at {self.clicks_made} clicks")
                self._click_cashout()
                
                payout = self.bet_amount * self.current_multiplier
                profit = payout - self.bet_amount
                self.current_profit += profit
                self.bankroll += profit
                self.wins += 1
                round_result = 'win'
                break
            
            # Select next tile
            tile = self._select_next_tile()
            if tile is None:
                logger.warning("No valid tiles, cashing out")
                self._click_cashout()
                round_result = 'win'
                break
            
            # Click the tile
            self._click_tile(tile[0], tile[1])
            self.clicks_made += 1
            
            # Wait for result
            time.sleep(0.5)
            
            # Update board state
            new_board_state = self._detect_board_state()
            if new_board_state is None:
                logger.error("Failed to detect board state after click")
                round_result = 'error'
                break
            
            # Check if mine was hit (tile still hidden but we clicked)
            if not new_board_state[tile[0], tile[1]]:
                # Possible mine hit
                logger.warning(f"Possible mine hit at ({tile[0]}, {tile[1]})")
                
                # Wait a bit more to confirm
                time.sleep(1.0)
                final_board_state = self._detect_board_state()
                
                if final_board_state and not final_board_state[tile[0], tile[1]]:
                    # Definitely a mine
                    logger.error("MINE HIT - Round lost")
                    profit = -self.bet_amount
                    self.current_profit += profit
                    self.bankroll += profit
                    self.losses += 1
                    round_result = 'loss'
                    break
            
            # Update multiplier (simulate calculation)
            self.current_multiplier = self._calculate_multiplier()
            self.board_state = new_board_state
            
            # Check if won (all safe tiles revealed)
            if np.sum(self.board_state) >= max_clicks:
                logger.info("All safe tiles revealed - Round won!")
                payout = self.bet_amount * self.current_multiplier
                profit = payout - self.bet_amount
                self.current_profit += profit
                self.bankroll += profit
                self.wins += 1
                round_result = 'win'
                break
        
        # Log round
        round_log = {
            'round': self.current_round,
            'clicks_made': self.clicks_made,
            'result': round_result,
            'profit': profit if round_result else 0,
            'bankroll': self.bankroll,
            'multiplier': self.current_multiplier
        }
        self.session_log.append(round_log)
        
        return round_log
    
    def _calculate_multiplier(self) -> float:
        """Calculate current multiplier based on clicks made."""
        # Use payout table
        payout_table = payout_table_25_2()
        return payout_table.get(self.clicks_made, 1.0)
    
    def run(self):
        """Run the bot for the configured number of rounds."""
        logger.info("="*70)
        logger.info("RAINBET MINES BOT - STARTING AUTOMATION")
        logger.info("="*70)
        logger.info(f"Character: {self.character_name}")
        logger.info(f"Bet Amount: ${self.bet_amount:.2f}")
        logger.info(f"Total Rounds: {self.total_rounds}")
        logger.info("="*70)
        
        # Calibrate first
        logger.info("Calibrating...")
        if not self.calibrate():
            logger.error("Calibration failed. Cannot proceed.")
            return
        
        logger.info("Calibration complete. Starting rounds in 5 seconds...")
        time.sleep(5)
        
        # Run rounds
        for round_num in range(self.total_rounds):
            if self.bankroll < self.bet_amount:
                logger.warning("Bankroll depleted. Stopping.")
                break
            
            round_log = self.run_round()
            logger.info(f"Round {round_num + 1}: {round_log.get('result', 'unknown')}")
            
            # Inter-round pause
            if self.human_emulator:
                pause_time = random.uniform(3, 8)
                logger.debug(f"Pausing for {pause_time:.1f}s")
                time.sleep(pause_time)
        
        # Final summary
        self._print_summary()
        self._save_results()
    
    def _print_summary(self):
        """Print final results summary."""
        win_rate = (self.wins / self.current_round * 100) if self.current_round > 0 else 0
        
        logger.info("="*70)
        logger.info("BOT SESSION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total Rounds: {self.current_round}")
        logger.info(f"Wins: {self.wins}, Losses: {self.losses}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Total Profit: ${self.current_profit:.2f}")
        logger.info(f"Final Bankroll: ${self.bankroll:.2f}")
        logger.info("="*70)
    
    def _save_results(self):
        """Save session results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"session_{timestamp}.json"
        
        results = {
            'character': self.character_name,
            'config': self.config,
            'total_rounds': self.current_round,
            'wins': self.wins,
            'losses': self.losses,
            'total_profit': self.current_profit,
            'final_bankroll': self.bankroll,
            'session_log': self.session_log
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")


def main():
    """Main entry point for the bot."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RainBet Mines Automation Bot')
    parser.add_argument('--character', choices=['takeshi', 'kazuya', 'lelouch', 'hybrid'],
                       default='hybrid', help='Character strategy to use')
    parser.add_argument('--bet', type=float, default=10.0, help='Bet amount')
    parser.add_argument('--rounds', type=int, default=100, help='Number of rounds')
    parser.add_argument('--bankroll', type=float, default=1000.0, help='Initial bankroll')
    parser.add_argument('--calibrate-only', action='store_true',
                       help='Only run calibration, then exit')
    
    args = parser.parse_args()
    
    config = {
        'character': args.character,
        'bet_amount': args.bet,
        'total_rounds': args.rounds,
        'initial_bankroll': args.bankroll,
        'anti_detection': True
    }
    
    bot = RainBetMinesBot(config)
    
    if args.calibrate_only:
        bot.calibrate()
    else:
        bot.run()


if __name__ == "__main__":
    main()

