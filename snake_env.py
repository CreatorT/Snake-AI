import pygame
import sys
import random
import os
import json
from collections import deque

class SnakeEnv:
    def __init__(self, render_mode=False, config_path="config.json"):
        pygame.init()
        self.render_mode = render_mode

        cfg = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as cfg_file:
                cfg = json.load(cfg_file)

        self.scale = cfg.get("SCALE", 1)
        self.WINDOW_WIDTH = cfg.get("WINDOW_WIDTH", 640) * self.scale
        self.WINDOW_HEIGHT = cfg.get("WINDOW_HEIGHT", 480) * self.scale
        self.CELL_SIZE = cfg.get("CELL_SIZE", 20) * self.scale
        self.GRID_WIDTH = self.WINDOW_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.WINDOW_HEIGHT // self.CELL_SIZE

        self.num_static_obstacle_clusters = cfg.get("NUM_STATIC_OBSTACLE_CLUSTERS", 3)
        self.cluster_max_size = cfg.get("CLUSTER_MAX_SIZE", 10)
        self.initial_moving_obstacles_count = cfg.get("INITIAL_MOVING_OBSTACLES_COUNT", 1)

        # Farben
        self.DARK_GREEN = (162, 209, 73)
        self.LIGHT_GREEN = (170, 215, 81)
        self.GREEN = (78, 124, 246)
        self.RED   = (231, 71, 29)
        self.WHITE = (255, 255, 255)
        self.GRAY  = (100, 100, 100)

        if self.render_mode:
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption("Snake RL")
        self.clock = pygame.time.Clock()
        self.base_FPS = cfg.get("BASE_FPS", 10)       #Faster speed #normal 10
        self.FPS = self.base_FPS

        # Textures (optional)
        self.snake_head_texture = None
        texture_filename = os.path.join("textures", "head.png")
        if os.path.exists(texture_filename):
            try:
                self.snake_head_texture = pygame.image.load(texture_filename).convert_alpha()
                self.snake_head_texture = pygame.transform.scale(self.snake_head_texture, (self.CELL_SIZE, self.CELL_SIZE))
            except Exception as e:
                print("Error loading snake head texture:", e)
                self.snake_head_texture = None

        self.apple_texture = None
        texture_filename = os.path.join("textures", "apple.png")
        if os.path.exists(texture_filename):
            try:
                self.apple_texture = pygame.image.load(texture_filename).convert_alpha()
                self.apple_texture = pygame.transform.scale(self.apple_texture, (self.CELL_SIZE, self.CELL_SIZE))
            except Exception as e:
                print("Error loading apple texture:", e)
                self.apple_texture = None

        self.stone_texture = None
        texture_filename = os.path.join("textures", "stone2.png")
        if os.path.exists(texture_filename):
            try:
                self.stone_texture = pygame.image.load(texture_filename).convert_alpha()
                self.stone_texture = pygame.transform.scale(self.stone_texture, (self.CELL_SIZE, self.CELL_SIZE))
            except Exception as e:
                print("Error loading stone texture:", e)
                self.stone_texture = None

        self.spike_texture = None
        texture_filename = os.path.join("textures", "spike.png")
        if os.path.exists(texture_filename):
            try:
                self.spike_texture = pygame.image.load(texture_filename).convert_alpha()
                self.spike_texture = pygame.transform.scale(self.spike_texture, (self.CELL_SIZE, self.CELL_SIZE))
            except Exception as e:
                print("Error loading spike texture:", e)
                self.spike_texture = None

        self.reset()

    def reset(self):
        # Initialisiere Spielzustand
        self.direction = "RIGHT"
        self.snake_pos = [self.CELL_SIZE * 6, self.CELL_SIZE * 3]
        self.snake_body = [
            [self.CELL_SIZE * 6, self.CELL_SIZE * 3],
            [self.CELL_SIZE * 5, self.CELL_SIZE * 3],
            [self.CELL_SIZE * 4, self.CELL_SIZE * 3],
            [self.CELL_SIZE * 3, self.CELL_SIZE * 3]
        ]
        self.score = 0
        self.level = 1
        self.FPS = self.base_FPS
        self.static_obstacles = set()
        self.moving_obstacles = []
        self.generate_static_obstacles(
            num_clusters=self.num_static_obstacle_clusters,
            cluster_max_size=self.cluster_max_size,
            growth_probability=0.5)
        for _ in range(self.initial_moving_obstacles_count):
            self.spawn_moving_obstacle()
        self.food_pos = self.get_random_food_position()
        self.food_spawn = True
        self.frame_iteration = 0
        return self.get_state()

    def step(self, action):
        """
        action: 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        """
        self.frame_iteration += 1

        action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        chosen_direction = action_map.get(action, self.direction)
        # Verhindere Richtungsumkehr
        if chosen_direction == "UP" and self.direction != "DOWN":
            self.direction = "UP"
        elif chosen_direction == "DOWN" and self.direction != "UP":
            self.direction = "DOWN"
        elif chosen_direction == "LEFT" and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif chosen_direction == "RIGHT" and self.direction != "LEFT":
            self.direction = "RIGHT"

        # Schlangenposition aktualisieren
        if self.direction == "UP":
            self.snake_pos[1] -= self.CELL_SIZE
        elif self.direction == "DOWN":
            self.snake_pos[1] += self.CELL_SIZE
        elif self.direction == "LEFT":
            self.snake_pos[0] -= self.CELL_SIZE
        elif self.direction == "RIGHT":
            self.snake_pos[0] += self.CELL_SIZE

        self.snake_body.insert(0, list(self.snake_pos))

        reward = 0
        done = False

        # Überprüfe food
        if self.snake_pos == self.food_pos:
            self.score += 1
            reward = 10
            self.food_spawn = False
        else:
            self.snake_body.pop()

        if not self.food_spawn:
            self.food_pos = self.get_random_food_position()
            self.food_spawn = True

        if self.is_collision(self.snake_pos):
            reward = -10
            done = True
            return self.get_state(), reward, done, {}

        self.update_moving_obstacles()
        for mob in self.moving_obstacles:
            for segment in self.snake_body[:4]:
                seg_cell = (segment[0] // self.CELL_SIZE, segment[1] // self.CELL_SIZE)
                if seg_cell == mob['pos']:
                    reward = -10
                    done = True
                    return self.get_state(), reward, done, {}

        # Level-Up: Alle 10 Punkte
        if self.score >= self.level * 10:
            self.level += 1
            self.FPS = self.base_FPS + self.level - 1
            self.spawn_moving_obstacle()

        if self.frame_iteration > 100 * len(self.snake_body):
            reward = -10
            done = True

        state = self.get_state()
        if self.render_mode:
            self.render()
        self.clock.tick(self.FPS)
        return state, reward, done, {}

    def is_collision(self, pos):
        x, y = pos
        # Kollision mit Wänden
        if x < 0 or x >= self.WINDOW_WIDTH or y < 0 or y >= self.WINDOW_HEIGHT:
            return True
        # Kollision mit sich selbst
        for segment in self.snake_body[1:]:
            if pos == segment:
                return True
        # Kollision mit statischen Hindernissen
        cell = (x // self.CELL_SIZE, y // self.CELL_SIZE)
        if cell in self.static_obstacles:
            return True
        return False

    def get_random_food_position(self):
        while True:
            x = random.randrange(0, self.WINDOW_WIDTH, self.CELL_SIZE)
            y = random.randrange(0, self.WINDOW_HEIGHT, self.CELL_SIZE)
            grid_pos = (x // self.CELL_SIZE, y // self.CELL_SIZE)
            if grid_pos not in self.static_obstacles and grid_pos not in [mob['pos'] for mob in self.moving_obstacles]:
                return [x, y]

    def render(self):
        self.draw_background()
        self.draw_static_obstacles()
        self.draw_moving_obstacles()
        food_rect = pygame.Rect(self.food_pos[0], self.food_pos[1], self.CELL_SIZE, self.CELL_SIZE)
        if self.apple_texture:
            self.screen.blit(self.apple_texture, food_rect)
        else:
            pygame.draw.rect(self.screen, self.RED, food_rect)
        for index, segment in enumerate(self.snake_body):
            rect = pygame.Rect(segment[0], segment[1], self.CELL_SIZE, self.CELL_SIZE)
            if index == 0 and self.snake_head_texture is not None:
                self.screen.blit(self.snake_head_texture, rect)
            else:
                pygame.draw.rect(self.screen, self.GREEN, rect)
        self.show_status(self.score, self.level)
        pygame.display.update()

    def get_state(self):
        head_x, head_y = self.snake_pos
        # Bestehende Features:
        dir_u = self.direction == "UP"
        dir_d = self.direction == "DOWN"
        dir_l = self.direction == "LEFT"
        dir_r = self.direction == "RIGHT"

        danger_straight = self.is_collision(self._move_point(self.snake_pos, self.direction))
        danger_right = self.is_collision(self._move_point(self.snake_pos, self._turn_right(self.direction)))
        danger_left = self.is_collision(self._move_point(self.snake_pos, self._turn_left(self.direction)))

        food_left = self.food_pos[0] < head_x
        food_right = self.food_pos[0] > head_x
        food_up = self.food_pos[1] < head_y
        food_down = self.food_pos[1] > head_y

        state = [
            1 if danger_straight else 0,
            1 if danger_right else 0,
            1 if danger_left else 0,
            1 if dir_u else 0,
            1 if dir_d else 0,
            1 if dir_l else 0,
            1 if dir_r else 0,
            1 if food_left else 0,
            1 if food_right else 0,
            1 if food_up else 0,
            1 if food_down else 0
        ]

        # Neue Features: Daten zum nächstgelegenen beweglichen Hindernis
        nearest_dx = 0
        nearest_dy = 0
        dir_mob = [0, 0, 0, 0]  # One-Hot: [UP, DOWN, LEFT, RIGHT]
        if self.moving_obstacles:
            # Finde das Hindernis, das am nächsten zum Schlangenkopf ist
            nearest = min(self.moving_obstacles, key=lambda mob: abs(mob['pos'][0]*self.CELL_SIZE - head_x) + abs(mob['pos'][1]*self.CELL_SIZE - head_y))
            mob_x = nearest['pos'][0] * self.CELL_SIZE
            mob_y = nearest['pos'][1] * self.CELL_SIZE
            # Berechne die Differenz (normalisiert durch Fenstergröße)
            nearest_dx = (mob_x - head_x) / self.WINDOW_WIDTH
            nearest_dy = (mob_y - head_y) / self.WINDOW_HEIGHT

            # Bestimme die Bewegungsrichtung des Hindernisses als One-Hot
            dx, dy = nearest['dir']
            # Annahme: (0,-1)=UP, (0,1)=DOWN, (-1,0)=LEFT, (1,0)=RIGHT
            if (dx, dy) == (0, -1):
                dir_mob = [1, 0, 0, 0]
            elif (dx, dy) == (0, 1):
                dir_mob = [0, 1, 0, 0]
            elif (dx, dy) == (-1, 0):
                dir_mob = [0, 0, 1, 0]
            elif (dx, dy) == (1, 0):
                dir_mob = [0, 0, 0, 1]
        # Falls kein Hindernis existiert, bleiben die neuen Features 0.
        state.extend([nearest_dx, nearest_dy])
        state.extend(dir_mob)

        return state


    def _move_point(self, point, direction):
        x, y = point
        if direction == "UP":
            return [x, y - self.CELL_SIZE]
        elif direction == "DOWN":
            return [x, y + self.CELL_SIZE]
        elif direction == "LEFT":
            return [x - self.CELL_SIZE, y]
        elif direction == "RIGHT":
            return [x + self.CELL_SIZE, y]
        return point

    def _turn_right(self, direction):
        mapping = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}
        return mapping[direction]

    def _turn_left(self, direction):
        mapping = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
        return mapping[direction]

    def draw_background(self):
        for y in range(0, self.WINDOW_HEIGHT, self.CELL_SIZE):
            for x in range(0, self.WINDOW_WIDTH, self.CELL_SIZE):
                color = self.DARK_GREEN if ((x // self.CELL_SIZE) + (y // self.CELL_SIZE)) % 2 == 0 else self.LIGHT_GREEN
                pygame.draw.rect(self.screen, color, pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE))

    def draw_static_obstacles(self):
        """Zeichnet alle statischen Hindernisse."""
        for cell in self.static_obstacles:
            x, y = cell[0] * self.CELL_SIZE, cell[1] * self.CELL_SIZE
            rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
            if self.stone_texture:
                self.screen.blit(self.stone_texture, rect)
            else:
                pygame.draw.rect(self.screen, self.GRAY, rect)

    def draw_moving_obstacles(self):
        """Zeichnet alle beweglichen Hindernisse."""
        for mob in self.moving_obstacles:
            x, y = mob['pos'][0] * self.CELL_SIZE, mob['pos'][1] * self.CELL_SIZE
            rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
            if self.spike_texture:
                self.screen.blit(self.spike_texture, rect)
            else:
                pygame.draw.rect(self.screen, self.GRAY, rect)

    def show_status(self, score, level):
        font = pygame.font.SysFont('consolas', 20)
        status_surface = font.render(f"Score: {score}  Level: {level}", True, self.WHITE)
        self.screen.blit(status_surface, (10, 10))

    def generate_static_obstacles(self, num_clusters=3, cluster_max_size=10, growth_probability=0.5):
        safe_zone = set()
        start_cell = (self.snake_body[0][0] // self.CELL_SIZE, self.snake_body[0][1] // self.CELL_SIZE)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = start_cell[0] + dx, start_cell[1] + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    safe_zone.add((nx, ny))
        while True:
            new_obstacles = set()
            for _ in range(num_clusters):
                while True:
                    center = (random.randrange(0, self.GRID_WIDTH), random.randrange(0, self.GRID_HEIGHT))
                    if center not in safe_zone:
                        break
                cluster = {center}
                frontier = [center]
                while frontier and len(cluster) < cluster_max_size:
                    cell = random.choice(frontier)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        neighbor = (cell[0] + dx, cell[1] + dy)
                        if (0 <= neighbor[0] < self.GRID_WIDTH and 0 <= neighbor[1] < self.GRID_HEIGHT and 
                            neighbor not in cluster and neighbor not in safe_zone):
                            if random.random() < growth_probability:
                                cluster.add(neighbor)
                                frontier.append(neighbor)
                    if cell in frontier:
                        frontier.remove(cell)
                new_obstacles = new_obstacles.union(cluster)
            if self.is_grid_fully_connected(new_obstacles, start_cell):
                self.static_obstacles = new_obstacles
                break

    def is_grid_fully_connected(self, obstacles_set, start):
        visited = set()
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in obstacles_set and (nx, ny) not in visited:
                        queue.append((nx, ny))
        free_cells = self.GRID_WIDTH * self.GRID_HEIGHT - len(obstacles_set)
        return len(visited) == free_cells

    def spawn_moving_obstacle(self):
        while True:
            cell = (random.randrange(0, self.GRID_WIDTH), random.randrange(0, self.GRID_HEIGHT))
            if cell not in self.static_obstacles:
                break
        dir_options = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        direction_vec = random.choice(dir_options)
        self.moving_obstacles.append({'pos': cell, 'dir': direction_vec})

    def update_moving_obstacles(self):
        for mob in self.moving_obstacles:
            x, y = mob['pos']
            dx, dy = mob['dir']
            new_x = x + dx
            new_y = y + dy
            if new_x < 0 or new_x >= self.GRID_WIDTH:
                dx = -dx
                new_x = x + dx
            if new_y < 0 or new_y >= self.GRID_HEIGHT:
                dy = -dy
                new_y = y + dy
            mob['pos'] = (new_x, new_y)
            mob['dir'] = (dx, dy)
