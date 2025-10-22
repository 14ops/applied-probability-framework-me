from collections import deque
import random
import time
import numpy as np


# Placeholder for a generic action function (e.g., clicking a cell)
def perform_action(action_name, delay_ms):
    # Simulate performing an action with a given delay
    # In a real system, this would be a call to a UI automation library (e.g., pyautogui, selenium)
    time.sleep(delay_ms / 1000.0)  # Convert ms to seconds
    # print(f"  Action: {action_name} performed after {delay_ms:.2f}ms")


class HumanBehaviorEmulator:
    def __init__(
        self,
        min_delay_ms=50,
        max_delay_ms=200,
        long_pause_interval=10,
        long_pause_min_s=5,
        long_pause_max_s=30,
    ):
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.long_pause_interval = (
            long_pause_interval  # Every N actions, a long pause might occur
        )
        self.long_pause_min_s = long_pause_min_s
        self.long_pause_max_s = long_pause_max_s
        self.action_count = 0

    def get_random_delay(self):
        # Randomized delays between actions
        delay = random.uniform(self.min_delay_ms, self.max_delay_ms)
        return delay

    def apply_human_jitter(self, action_name):
        delay = self.get_random_delay()
        perform_action(action_name, delay)
        self.action_count += 1

        # Occasional \'Patience\' Pauses
        if (
            self.action_count % self.long_pause_interval == 0 and random.random() < 0.3
        ):  # 30% chance for a long pause
            long_pause_duration = random.uniform(
                self.long_pause_min_s, self.long_pause_max_s
            )
            print(f"  (Human-like pause for {long_pause_duration:.2f}s)")
            time.sleep(long_pause_duration)

    def simulate_mouse_drift(self, start_pos, end_pos, num_steps=5):
        # This is conceptual for a simulation. In a real UI, this would involve pyautogui.moveTo with random offsets.
        # print(f"  Simulating mouse drift from {start_pos} to {end_pos}")
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            # Linear interpolation
            current_x = start_pos[0] * (1 - t) + end_pos[0] * t
            current_y = start_pos[1] * (1 - t) + end_pos[1] * t

            # Add random jitter to intermediate points
            if i > 0 and i < num_steps:
                jitter_x = random.uniform(-2, 2)
                jitter_y = random.uniform(-2, 2)
                current_x += jitter_x
                current_y += jitter_y
            path.append((current_x, current_y))
        return path


class DetectionSimulator:
    def __init__(self, aggressiveness=0.5):
        self.aggressiveness = aggressiveness  # 0.0 (lenient) to 1.0 (highly aggressive)
        self.detection_score = 0.0
        self.last_action_time = time.time()
        self.action_timings = deque(maxlen=10)  # Keep track of last 10 action timings
        self.mouse_movements = deque(
            maxlen=5
        )  # Keep track of last 5 mouse movements (conceptual)

    def _evaluate_timing_consistency(self, current_time):
        time_diff = current_time - self.last_action_time
        self.action_timings.append(time_diff)
        self.last_action_time = current_time

        if len(self.action_timings) < 5:  # Need enough data points
            return 0.0

        # Check for unusually precise or repetitive timings
        std_dev = np.std(list(self.action_timings))
        mean_timing = np.mean(list(self.action_timings))

        # If std_dev is very low (too consistent) or timings are too fast/slow
        if std_dev < 0.01 * mean_timing:  # Very low standard deviation
            return 0.3 * self.aggressiveness  # Higher score for robotic consistency
        if mean_timing < 0.05:  # Too fast (less than 50ms average)
            return 0.2 * self.aggressiveness
        return 0.0

    def _evaluate_mouse_path(self, mouse_path):
        # Conceptual: In a real scenario, this would analyze actual mouse coordinates
        # For simulation, we assume mouse_path is a list of (x,y) tuples
        if not mouse_path or len(mouse_path) < 2:
            return 0.0

        # Check for perfectly straight lines (lack of jitter)
        # Simplified: if all points are on a perfect line, score higher
        # More complex: analyze curvature, speed changes, etc.
        straightness_score = 0.0
        if len(mouse_path) > 2:
            # Check if intermediate points deviate from line between start and end
            start = mouse_path[0]
            end = mouse_path[-1]
            # Calculate distance from point p to line segment (start, end)
            # Simplified check: if p is exactly on the line
            for i in range(1, len(mouse_path) - 1):
                p = mouse_path[i]  # Assign p here
                if (
                    abs(
                        (end[1] - start[1]) * p[0]
                        - (end[0] - start[0]) * p[1]
                        + end[0] * start[1]
                        - end[1] * start[0]
                    )
                    < 0.1
                ):
                    straightness_score += 0.1
            if straightness_score > 0.5:  # If many points are too straight
                return 0.4 * self.aggressiveness
        return 0.0

    def update_detection_score(self, action_type, action_details=None):
        current_time = time.time()
        score_increase = 0.0

        # Evaluate timing consistency
        score_increase += self._evaluate_timing_consistency(current_time)

        # Evaluate mouse path (if applicable)
        if (
            action_type == "mouse_click"
            and action_details
            and "mouse_path" in action_details
        ):
            score_increase += self._evaluate_mouse_path(action_details["mouse_path"])

        # Add other heuristics (e.g., rapid, repetitive actions without pause)
        # For now, a simple decay for the score
        self.detection_score = max(
            0, self.detection_score * 0.95 + score_increase
        )  # Decay over time
        return self.detection_score

    def is_detected(self, threshold=0.7):
        return self.detection_score >= threshold


# Example Usage
if __name__ == "__main__":
    emulator = HumanBehaviorEmulator(
        min_delay_ms=100,
        max_delay_ms=500,
        long_pause_interval=5,
        long_pause_min_s=2,
        long_pause_max_s=10,
    )
    detector = DetectionSimulator(aggressiveness=0.7)

    print("\n--- Anti-Detection Simulation ---")
    num_simulated_actions = 20
    current_mouse_pos = (0, 0)

    for i in range(num_simulated_actions):
        print(f"\nAction {i+1}/{num_simulated_actions}")
        action_type = random.choice(["mouse_click", "keyboard_input"])
        action_details = {}

        if action_type == "mouse_click":
            target_pos = (random.randint(10, 900), random.randint(10, 700))
            mouse_path = emulator.simulate_mouse_drift(current_mouse_pos, target_pos)
            action_details["mouse_path"] = mouse_path
            current_mouse_pos = target_pos
            emulator.apply_human_jitter("Click")
        else:
            emulator.apply_human_jitter("Type")

        detection_score = detector.update_detection_score(action_type, action_details)
        print(f"  Current Detection Score: {detection_score:.2f}")

        if detector.is_detected():
            print(f"!!! DETECTED at action {i+1}! Time to detection: {i+1} actions.")
            break

    if not detector.is_detected():
        print(
            "\nSimulation complete. Bot remained undetected. Keep up the stealth, ninja!"
        )
    else:
        print(
            "\nSimulation complete. Detection occurred. Time to refine your disguise, agent!"
        )

    print(
        "\nThis simulation demonstrates how to model human-like behavior and test against detection heuristics. It\\'s like a spy training against a simulated security system to perfect their infiltration techniques!"
    )
