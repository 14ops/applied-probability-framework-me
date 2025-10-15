
import pytest
import random
from collections import deque

# Assuming GameSimulator and BasicStrategy are in the src directory
# For testing purposes, we'll include simplified versions or mock them

class GameSimulator:
    def __init__(self, board_size=5, mine_count=5):
        self.board_size = board_size
        self.mine_count = mine_count
        self.board = []
        self.revealed = []
        self.mines = set()
        self.current_payout = 1.0
        self.game_over = False
        self.win = False
        self.cells_clicked = 0

    def initialize_game(self):
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.revealed = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.mines = set()
        self.current_payout = 1.0
        self.game_over = False
        self.win = False
        self.cells_clicked = 0
        
        all_cells = [(r, c) for r in range(self.board_size) for c in range(self.board_size)]
        self.mines = set(random.sample(all_cells, self.mine_count))

    def click_cell(self, r, c):
        if self.game_over or self.revealed[r][c]:
            return False, "Invalid click"

        self.revealed[r][c] = True
        self.cells_clicked += 1

        if (r, c) in self.mines:
            self.game_over = True
            self.win = False
            return True, "Mine"
        else:
            self.current_payout *= 1.1
            non_mine_cells_to_reveal = (self.board_size * self.board_size) - self.mine_count
            if self.cells_clicked == non_mine_cells_to_reveal:
                self.game_over = True
                self.win = True
            return True, "Safe"

    def cash_out(self):
        if not self.game_over:
            self.game_over = True
            self.win = True
            return True, "Cashed out"
        return False, "Game already over"


# Unit tests for GameSimulator
def test_game_initialization():
    simulator = GameSimulator(board_size=5, mine_count=3)
    simulator.initialize_game()
    assert simulator.board_size == 5
    assert simulator.mine_count == 3
    assert len(simulator.mines) == 3
    assert simulator.current_payout == 1.0
    assert not simulator.game_over
    assert not simulator.win
    assert simulator.cells_clicked == 0

def test_click_safe_cell():
    simulator = GameSimulator(board_size=3, mine_count=1)
    simulator.initialize_game()
    # Ensure we pick a cell that is not a mine for this test
    safe_cell = None
    for r in range(simulator.board_size):
        for c in range(simulator.board_size):
            if (r, c) not in simulator.mines:
                safe_cell = (r, c)
                break
        if safe_cell: break

    if safe_cell:
        r, c = safe_cell
        result, message = simulator.click_cell(r, c)
        assert result is True
        assert message == "Safe"
        assert simulator.revealed[r][c] is True
        assert simulator.cells_clicked == 1
        assert simulator.current_payout > 1.0
        assert not simulator.game_over

def test_click_mine_cell():
    simulator = GameSimulator(board_size=3, mine_count=1)
    simulator.initialize_game()
    mine_cell = list(simulator.mines)[0]
    r, c = mine_cell
    result, message = simulator.click_cell(r, c)
    assert result is True
    assert message == "Mine"
    assert simulator.revealed[r][c] is True
    assert simulator.game_over is True
    assert simulator.win is False

def test_cash_out():
    simulator = GameSimulator(board_size=3, mine_count=1)
    simulator.initialize_game()
    # Click a safe cell first to make it a valid game state
    safe_cell = None
    for r in range(simulator.board_size):
        for c in range(simulator.board_size):
            if (r, c) not in simulator.mines:
                safe_cell = (r, c)
                break
        if safe_cell: break
    
    if safe_cell:
        simulator.click_cell(safe_cell[0], safe_cell[1])

    result, message = simulator.cash_out()
    assert result is True
    assert message == "Cashed out"
    assert simulator.game_over is True
    assert simulator.win is True

def test_win_by_clearing_all_safe_cells():
    simulator = GameSimulator(board_size=2, mine_count=1) # 3 safe cells
    simulator.initialize_game()
    safe_cells = []
    for r in range(simulator.board_size):
        for c in range(simulator.board_size):
            if (r, c) not in simulator.mines:
                safe_cells.append((r, c))
    
    # Click all safe cells
    for r, c in safe_cells:
        simulator.click_cell(r, c)
    
    assert simulator.game_over is True
    assert simulator.win is True
    assert simulator.cells_clicked == len(safe_cells)


# Placeholder for a simple strategy to test integration
class BasicStrategy:
    def __init__(self):
        self.name = "BasicStrategy"

    def apply(self, simulator):
        unrevealed_cells = []
        for r in range(simulator.board_size):
            for c in range(simulator.board_size):
                if not simulator.revealed[r][c]:
                    unrevealed_cells.append((r, c))
        
        if unrevealed_cells:
            r, c = random.choice(unrevealed_cells)
            simulator.click_cell(r, c)
            if simulator.cells_clicked > 1 and random.random() < 0.3:
                simulator.cash_out()
        else:
            if not simulator.game_over:
                simulator.cash_out()

def test_basic_strategy_applies():
    simulator = GameSimulator(board_size=3, mine_count=1)
    simulator.initialize_game()
    strategy = BasicStrategy()
    strategy.apply(simulator)
    # The strategy should have performed some actions, so game_over should be True or cells_clicked > 0
    assert simulator.game_over or simulator.cells_clicked > 0


