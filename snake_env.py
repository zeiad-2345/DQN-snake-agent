import numpy as np
import random


class SnakeEnv:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        self.direction = (0, 1)  # moving right (row_delta, col_delta)
        self.food = self._place_food()
        self.steps = 0
        self.max_steps = 100 * len(self.snake)
        return self._get_state(), {}

    def _place_food(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in self.snake:
                return pos

    def _get_state(self):
        head = self.snake[0]
        dr, dc = self.direction
        right_dir = (dc, -dr)
        left_dir = (-dc, dr)

        def is_danger(d):
            r, c = head[0] + d[0], head[1] + d[1]
            return r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size or (r, c) in self.snake[1:]

        food_r, food_c = self.food
        state = [
            int(is_danger((dr, dc))),      # danger straight
            int(is_danger(right_dir)),     # danger right
            int(is_danger(left_dir)),      # danger left
            int(dr == 0 and dc == -1),     # moving left
            int(dr == 0 and dc == 1),      # moving right
            int(dr == -1 and dc == 0),     # moving up
            int(dr == 1 and dc == 0),      # moving down
            int(food_c < head[1]),         # food left
            int(food_c > head[1]),         # food right
            int(food_r < head[0]),         # food up
            int(food_r > head[0]),         # food down
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        # action: 0=straight, 1=turn right, 2=turn left
        dr, dc = self.direction
        if action == 1:
            self.direction = (dc, -dr)
        elif action == 2:
            self.direction = (-dc, dr)

        dr, dc = self.direction
        new_head = (self.snake[0][0] + dr, self.snake[0][1] + dc)
        self.steps += 1

        terminated = (
            new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake[1:]
        )
        if terminated:
            return self._get_state(), -10.0, True, False, {}

        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 10.0
            self.food = self._place_food()
            self.max_steps = 100 * len(self.snake)
        else:
            self.snake.pop()
            reward = 0.0

        truncated = self.steps >= self.max_steps
        return self._get_state(), reward, False, truncated, {}

    @property
    def observation_space(self):
        class _Space:
            shape = (11,)
        return _Space()

    @property
    def action_space(self):
        class _Space:
            n = 3
        return _Space()

    def close(self):
        pass
