"""Unit tests for the GameSimulator class."""

import pytest
from ..game_simulator import GameSimulator  # Corrected relative import


class TestGameSimulator:
    """Tests various functionalities of the GameSimulator."""

    def setup_method(self):
        """Set up a new GameSimulator instance before each test."""
        self.simulator = GameSimulator(board_size=5, mine_count=3, seed=42)

    def test_initialization(self):
        """Test if the simulator initializes correctly."""
        assert self.simulator.board_size == 5
        assert self.simulator.mine_count == 3
        assert not self.simulator.game_over
        assert not self.simulator.win
        assert self.simulator.cells_clicked == 0
        assert self.simulator.current_payout == 1.0
        assert len(self.simulator.mines) == 3
        assert len(self.simulator.board) == 5
        assert len(self.simulator.board[0]) == 5

    def test_initialize_game_resets_state(self):
        """Test if initialize_game resets the game state properly."""
        self.simulator.click_cell(0, 0)
        self.simulator.game_over = True
        self.simulator.win = True
        self.simulator.current_payout = 5.0
        self.simulator.cells_clicked = 10

        self.simulator.initialize_game()

        assert not self.simulator.game_over
        assert not self.simulator.win
        assert self.simulator.cells_clicked == 0
        assert self.simulator.current_payout == 1.0
        assert len(self.simulator.mines) == 3

    def test_click_safe_cell(self):
        """Test clicking a safe cell."""
        # Ensure (0,0) is safe for this test by re-initializing if it's a mine
        while (0, 0) in self.simulator.mines:
            self.simulator.initialize_game()

        initial_payout = self.simulator.current_payout
        result, message = self.simulator.click_cell(0, 0)

        assert result is True
        assert message == "Safe"
        assert self.simulator.revealed[0][0] is True
        assert self.simulator.current_payout == initial_payout * 1.1
        assert self.simulator.cells_clicked == 1
        assert not self.simulator.game_over

    def test_click_mine_cell(self):
        """Test clicking a mine cell."""
        # Ensure (0,0) is a mine for this test by re-initializing if it's safe
        while (0, 0) not in self.simulator.mines:
            self.simulator.initialize_game()

        result, message = self.simulator.click_cell(0, 0)

        assert result is True
        assert message == "Mine"
        assert self.simulator.revealed[0][0] is True
        assert self.simulator.game_over is True
        assert self.simulator.win is False

    def test_click_already_revealed_cell(self):
        """Test clicking a cell that has already been revealed."""
        # Ensure (0,0) is safe
        while (0, 0) in self.simulator.mines:
            self.simulator.initialize_game()

        self.simulator.click_cell(0, 0)
        result, message = self.simulator.click_cell(0, 0)

        assert result is False
        assert message == "Invalid click"
        assert self.simulator.cells_clicked == 1  # Should not increment again

    def test_cash_out(self):
        """Test the cash_out functionality."""
        # Ensure (0,0) is safe
        while (0, 0) in self.simulator.mines:
            self.simulator.initialize_game()

        self.simulator.click_cell(0, 0)
        initial_payout = self.simulator.current_payout
        result, message = self.simulator.cash_out()

        assert result is True
        assert message == "Cashed out"
        assert self.simulator.game_over is True
        assert self.simulator.win is True
        assert self.simulator.current_payout == initial_payout

    def test_win_condition(self):
        """Test the win condition when all non-mine cells are clicked."""
        # Create a tiny board with 1 mine and 2 non-mine cells
        self.simulator = GameSimulator(board_size=2, mine_count=1, seed=42)
        self.simulator.initialize_game()

        # Find the non-mine cells
        non_mine_cells = [
            (r, c)
            for r in range(self.simulator.board_size)
            for c in range(self.simulator.board_size)
            if (r, c) not in self.simulator.mines
        ]

        # Click all non-mine cells
        for r, c in non_mine_cells:
            self.simulator.click_cell(r, c)

        assert self.simulator.game_over is True
        assert self.simulator.win is True
        # Payout should be (1.1)^(number of non-mine cells)
        assert self.simulator.current_payout == round(1.1 ** len(non_mine_cells), 5)

    def test_game_over_prevents_further_clicks(self):
        """Test that no more clicks are allowed after game over."""
        # Ensure (0,0) is a mine
        while (0, 0) not in self.simulator.mines:
            self.simulator.initialize_game()

        self.simulator.click_cell(0, 0)  # Click a mine, game over
        assert self.simulator.game_over is True

        result, message = self.simulator.click_cell(1, 1)  # Try to click another cell
        assert result is False
        assert message == "Invalid click"

    def test_payout_progression(self):
        """Test the payout progression for safe clicks."""
        # Ensure (0,0), (0,1), (0,2) are safe
        safe_cells = [(0, 0), (0, 1), (0, 2)]
        self.simulator = GameSimulator(board_size=3, mine_count=0, seed=42)
        self.simulator.initialize_game()

        # Click safe cells and check payout
        payout = 1.0
        for r, c in safe_cells:
            self.simulator.click_cell(r, c)
            payout *= 1.1
            assert self.simulator.current_payout == round(payout, 5)

    def test_random_seed_reproducibility(self):
        """Test if the random seed ensures reproducibility."""
        sim1 = GameSimulator(board_size=5, mine_count=3, seed=123)
        sim1.initialize_game()
        mines1 = sorted(list(sim1.mines))

        sim2 = GameSimulator(board_size=5, mine_count=3, seed=123)
        sim2.initialize_game()
        mines2 = sorted(list(sim2.mines))

        assert mines1 == mines2

        sim1.click_cell(0, 0)
        sim2.click_cell(0, 0)
        assert sim1.game_over == sim2.game_over
        assert sim1.current_payout == sim2.current_payout

    def test_too_many_mines_or_cells(self):
        """Test edge cases for mine count and board size."""
        with pytest.raises(ValueError):
            GameSimulator(board_size=1, mine_count=2)  # More mines than cells
        with pytest.raises(ValueError):
            GameSimulator(board_size=0, mine_count=0)  # Zero board size


