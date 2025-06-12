# Standard imports
import pygame
import sys
import random
import os
import math
import json
import pygame.gfxdraw
from collections import deque
import torch
# ----------------- Original Game Configuration and Functions ------------------

with open("config.json", "r") as cfg_file:
    cfg = json.load(cfg_file)

USE_LIVES = cfg.get("USE_LIVES", True)
INITIAL_LIVES = cfg.get("INITIAL_LIVES", 3)
USE_WALLS = cfg.get("USE_WALLS", False)
USE_STATIC_OBSTACLES = cfg.get("USE_STATIC_OBSTACLES", True)
NUM_STATIC_OBSTACLE_CLUSTERS = cfg.get("NUM_STATIC_OBSTACLE_CLUSTERS", 3)
CLUSTER_MAX_SIZE = cfg.get("CLUSTER_MAX_SIZE", 10)
USE_MOVING_OBSTACLES = cfg.get("USE_MOVING_OBSTACLES", True)
INITIAL_MOVING_OBSTACLES_COUNT = cfg.get("INITIAL_MOVING_OBSTACLES_COUNT", 1)
NUM_APPLES = cfg.get("NUM_APPLES", 1)
BOOST_ENABLED = cfg.get("BOOST_ENABLED", True)
BOOST_COOLDOWN = cfg.get("BOOST_COOLDOWN", 5000)
BOOST_DISTANCE = cfg.get("BOOST_DISTANCE", 5)
INVULNERABILITY_TIME = cfg.get("INVULNERABILITY_TIME", 2000)
INPUT_INVERTED = cfg.get("INPUT_INVERTED", False)

pygame.init()
Scale = cfg.get("SCALE", 2)
base_width = cfg.get("WINDOW_WIDTH", 640)
base_height = cfg.get("WINDOW_HEIGHT", 480)
WINDOW_WIDTH = base_width * Scale
WINDOW_HEIGHT = base_height * Scale
CELL_SIZE = cfg.get("CELL_SIZE", 16) * Scale

DARK_GREEN = (162, 209, 73)
LIGHT_GREEN = (170, 215, 81)
GREEN = (78, 124, 246)
RED = (231, 71, 29)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)

GRID_WIDTH = WINDOW_WIDTH // CELL_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // CELL_SIZE

direction = "RIGHT"
change_to = direction
head_direction = 0

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Serpent of the Ancients")

clock = pygame.time.Clock()
base_FPS = cfg.get("BASE_FPS", 10)
FPS = base_FPS

snake_head_texture = None
texture_filename = os.path.join('textures', 'head.png')
if os.path.exists(texture_filename):
    snake_head_texture = pygame.image.load(texture_filename).convert_alpha()
    snake_head_texture = pygame.transform.scale_by(snake_head_texture, Scale)

apple_texture = None
texture_filename = os.path.join('textures', 'apple.png')
if os.path.exists(texture_filename):
    apple_texture = pygame.image.load(texture_filename).convert_alpha()
    apple_texture = pygame.transform.scale_by(apple_texture, Scale)

stone_texture = None
texture_filename = os.path.join('textures', 'stone2.png')
if os.path.exists(texture_filename):
    stone_texture = pygame.image.load(texture_filename).convert_alpha()
    stone_texture = pygame.transform.scale_by(stone_texture, Scale)

spike_texture = None
texture_filename = os.path.join('textures', 'spike.png')
if os.path.exists(texture_filename):
    spike_texture = pygame.image.load(texture_filename).convert_alpha()
    spike_texture = pygame.transform.scale_by(spike_texture, Scale)

INITIAL_SNAKE_POS = [CELL_SIZE * 6, CELL_SIZE * 3]
snake_pos = INITIAL_SNAKE_POS.copy()
snake_body = [
    [CELL_SIZE * 6, CELL_SIZE * 3],
    [CELL_SIZE * 5, CELL_SIZE * 3],
    [CELL_SIZE * 4, CELL_SIZE * 3],
    [CELL_SIZE * 3, CELL_SIZE * 3]
]

score = 0
level = 1
lives = INITIAL_LIVES if USE_LIVES else 1
invulnerable_until = 0
static_obstacles = set()
moving_obstacles = []
food_positions = []
last_boost_time = -BOOST_COOLDOWN

# ----------------- New UI Classes -----------------

# --- Button Class for interactive elements ---
class Button:
    def __init__(self, rect, text, font, base_color, hover_color, text_color, border_color, border_width=2, border_radius=8, callback=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.base_color = base_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_color = border_color
        self.border_width = border_width
        self.border_radius = border_radius
        self.callback = callback
        self.hovered = False

    def draw(self, surface):
        color = self.hover_color if self.hovered else self.base_color
        pygame.draw.rect(surface, color, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, self.border_color, self.rect, width=self.border_width, border_radius=self.border_radius)
        text_surf = self.font.render(self.text, True, self.text_color)
        surface.blit(text_surf, (self.rect.centerx - text_surf.get_width() // 2,
                                 self.rect.centery - text_surf.get_height() // 2))
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered and self.callback:
                self.callback()

# --- Options Menu ---
class OptionsMenu:
    def __init__(self, rect):
        """
        :param rect: Rectangle of the options panel (x, y, width, height)
        """
        self.rect = pygame.Rect(rect)
        self.bg_color = (60, 60, 60, 220)  # slightly transparent
        self.panel_border_color = (220, 220, 220)
        self.font = pygame.font.SysFont(None, 28)
        self.header_font = pygame.font.SysFont(None, 36, bold=True)
        self.button_font = pygame.font.SysFont(None, 26)
        
        self.highscore = 12345  # example value

        self.padding = 20
        self.spacing = 15
        
        self.continue_button = Button(
            rect=(self.padding, self.rect.height - 50, 120, 35),
            text="Continue",
            font=self.button_font,
            base_color=(80, 160, 80),
            hover_color=(100, 200, 100),
            text_color=(255, 255, 255),
            border_color=(220, 220, 220),
            callback=self.continue_game
        )
        
        self.quit_button = Button(
            rect=(self.rect.width - 120 - self.padding, self.rect.height - 50, 120, 35),
            text="Quit",
            font=self.button_font,
            base_color=(160, 80, 80),
            hover_color=(200, 100, 100),
            text_color=(255, 255, 255),
            border_color=(220, 220, 220),
            callback=self.quit_game
        )

        self.continued = False


    def continue_game(self):
        print("Continue selected")
        self.continued = True

    def quit_game(self):
        pygame.quit()
        sys.exit()

    def draw(self, surface):
        panel_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        panel_surf.fill(self.bg_color)
        pygame.draw.rect(panel_surf, self.bg_color, panel_surf.get_rect(), border_radius=15)
        pygame.draw.rect(panel_surf, self.panel_border_color, panel_surf.get_rect(), width=2, border_radius=15)
        
        header_text = self.header_font.render("Options", True, self.panel_border_color)
        panel_surf.blit(header_text, (self.rect.width // 2 - header_text.get_width() // 2, 10))
        
        
        highscore_text = self.font.render("Highscore: " + str(self.highscore), True, self.panel_border_color)
        panel_surf.blit(highscore_text, (20, 65 + 2 * (35 + self.spacing)))
        
        self.continue_button.draw(panel_surf)
        self.quit_button.draw(panel_surf)
        
        surface.blit(panel_surf, (self.rect.x, self.rect.y))
    
    def handle_event(self, event):
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            local_pos = (event.pos[0] - self.rect.x, event.pos[1] - self.rect.y)
            adjusted_event = pygame.event.Event(event.type, {"pos": local_pos})
            self.continue_button.handle_event(adjusted_event)
            self.quit_button.handle_event(adjusted_event)
        else:
            self.continue_button.handle_event(event)
            self.quit_button.handle_event(event)


# --- RoundAnimatedButton (Boost Button) ---
class RoundAnimatedButton:
    def __init__(self, center, radius, bg_color, border_color, border_width=3, load_time=5.0):
        """
        :param center: Center of the button (x, y)
        :param radius: Radius of the button
        :param bg_color: Background color as an RGBA tuple, e.g., (255, 255, 255, 100)
        :param border_color: Color of the animated border (RGB)
        :param border_width: Width of the border
        :param load_time: Time in seconds for the loading animation to complete (not used directly anymore)
        """
        self.center = center
        self.radius = radius
        self.bg_color = bg_color
        self.border_color = border_color
        self.border_width = border_width
        self.load_time = load_time
        self.progress = 0.0  # Progress from 0.0 (empty) to 1.0 (full)
        self.hovered = False
        self.pressed = False

    def update(self, dt, current_time, last_boost_time, cooldown):
        # Calculate progress based on time elapsed since the last boost
        self.progress = min((current_time - last_boost_time) / cooldown, 1.0)
        mouse_pos = pygame.mouse.get_pos()
        dx = mouse_pos[0] - self.center[0]
        dy = mouse_pos[1] - self.center[1]
        self.hovered = (math.hypot(dx, dy) <= self.radius)

    def draw(self, surface):
        button_surf = pygame.Surface((2*self.radius, 2*self.radius), pygame.SRCALPHA)
        
        scale = 0.95 if self.pressed else 1.0
        r = int(self.radius * scale)
        center_local = (self.radius, self.radius)
        
        if self.hovered:
            adjusted_alpha = min(self.bg_color[3] + 50, 255)
        else:
            adjusted_alpha = self.bg_color[3]
        adjusted_bg_color = (self.bg_color[0], self.bg_color[1], self.bg_color[2], adjusted_alpha)
        
        pygame.gfxdraw.filled_circle(button_surf, center_local[0], center_local[1], r, adjusted_bg_color)
        pygame.gfxdraw.aacircle(button_surf, center_local[0], center_local[1], r, adjusted_bg_color)
        
        rect = pygame.Rect(center_local[0] - r, center_local[1] - r, 2*r, 2*r)
        start_angle = -math.pi / 2
        
        if self.progress < 1.0:
            end_angle = start_angle + self.progress * 2 * math.pi
            pygame.draw.arc(button_surf, self.border_color, rect, start_angle, end_angle, self.border_width)
        else:
            pygame.gfxdraw.aacircle(button_surf, center_local[0], center_local[1], r, self.border_color)
            pygame.draw.circle(button_surf, self.border_color, center_local, r, self.border_width)
        
        top_left = (self.center[0] - self.radius, self.center[1] - self.radius)
        surface.blit(button_surf, top_left)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            dx = mouse_pos[0] - self.center[0]
            dy = mouse_pos[1] - self.center[1]
            if math.hypot(dx, dy) <= self.radius:
                self.pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.pressed = False



def show_status(score, level, lives):
    font = pygame.font.SysFont('consolas', 20)
    status_surface = font.render(f"Score: {score}  Level: {level}", True, WHITE)
    screen.blit(status_surface, (10, 10))
    if USE_LIVES:
        lives_surface = font.render("Lives: " + "♥" * lives, True, RED)
        screen.blit(lives_surface, (WINDOW_WIDTH - lives_surface.get_width() - 10, 10))

def draw_background():
    for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            color = DARK_GREEN if ((x // CELL_SIZE) + (y // CELL_SIZE)) % 2 == 0 else LIGHT_GREEN
            pygame.draw.rect(screen, color, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))

def direction_to_vector(dir_str):
    if dir_str == "UP":
        return (0, -1)
    elif dir_str == "DOWN":
        return (0, 1)
    elif dir_str == "LEFT":
        return (-1, 0)
    elif dir_str == "RIGHT":
        return (1, 0)
    return (0, 0)

def _move_point(point, direction):
    x, y = point
    if direction == "UP":
        return [x, y - CELL_SIZE]
    elif direction == "DOWN":
        return [x, y + CELL_SIZE]
    elif direction == "LEFT":
        return [x - CELL_SIZE, y]
    elif direction == "RIGHT":
        return [x + CELL_SIZE, y]
    return [x, y]

def _turn_right(direction):
    mapping = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}
    return mapping[direction]

def _turn_left(direction):
    mapping = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
    return mapping[direction]

def is_grid_fully_connected(obstacles_set, start):
    visited = set()
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                if (nx, ny) not in obstacles_set and (nx, ny) not in visited:
                    queue.append((nx, ny))
    free_cells = GRID_WIDTH * GRID_HEIGHT - len(obstacles_set)
    return len(visited) == free_cells

def generate_static_obstacles(num_clusters=NUM_STATIC_OBSTACLE_CLUSTERS, cluster_max_size=CLUSTER_MAX_SIZE, growth_probability=0.5):
    global static_obstacles
    if not USE_STATIC_OBSTACLES:
        static_obstacles = set()
        return
    safe_zone = set()
    start_cell = (snake_body[0][0] // CELL_SIZE, snake_body[0][1] // CELL_SIZE)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = start_cell[0] + dx, start_cell[1] + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                safe_zone.add((nx, ny))
    while True:
        new_obstacles = set()
        for _ in range(num_clusters):
            while True:
                center = (random.randrange(0, GRID_WIDTH), random.randrange(0, GRID_HEIGHT))
                if center not in safe_zone:
                    break
            cluster = {center}
            frontier = [center]
            while frontier and len(cluster) < cluster_max_size:
                cell = random.choice(frontier)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (cell[0] + dx, cell[1] + dy)
                    if (0 <= neighbor[0] < GRID_WIDTH and 0 <= neighbor[1] < GRID_HEIGHT and 
                        neighbor not in cluster and neighbor not in safe_zone):
                        if random.random() < growth_probability:
                            cluster.add(neighbor)
                            frontier.append(neighbor)
                if cell in frontier:
                    frontier.remove(cell)
            new_obstacles = new_obstacles.union(cluster)
        if is_grid_fully_connected(new_obstacles, start_cell):
            static_obstacles = new_obstacles
            break

def draw_static_obstacles():
    for cell in static_obstacles:
        x, y = cell[0] * CELL_SIZE, cell[1] * CELL_SIZE
        if stone_texture:
            screen.blit(stone_texture, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))
        else:
            pygame.draw.rect(screen, GRAY, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))

def spawn_moving_obstacle():
    if not USE_MOVING_OBSTACLES:
        return
    while True:
        cell = (random.randrange(0, GRID_WIDTH), random.randrange(0, GRID_HEIGHT))
        if cell not in static_obstacles:
            break
    direction_vec = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    moving_obstacles.append({'pos': cell, 'dir': direction_vec})

def update_moving_obstacles():
    if not USE_MOVING_OBSTACLES:
        return
    for mob in moving_obstacles:
        x, y = mob['pos']
        dx, dy = mob['dir']
        new_x = x + dx
        new_y = y + dy
        if new_x < 0 or new_x >= GRID_WIDTH:
            dx = -dx
            new_x = x + dx
        if new_y < 0 or new_y >= GRID_HEIGHT:
            dy = -dy
            new_y = y + dy
        mob['pos'] = (new_x, new_y)
        mob['dir'] = (dx, dy)

def draw_moving_obstacles():
    if not USE_MOVING_OBSTACLES:
        return
    for mob in moving_obstacles:
        x, y = mob['pos'][0] * CELL_SIZE, mob['pos'][1] * CELL_SIZE
        if spike_texture:
            screen.blit(spike_texture, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))
        else:
            pygame.draw.rect(screen, GRAY, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))

def get_random_food_position():
    while True:
        x = random.randrange(0, WINDOW_WIDTH, CELL_SIZE)
        y = random.randrange(0, WINDOW_HEIGHT, CELL_SIZE)
        grid_pos = (x // CELL_SIZE, y // CELL_SIZE)
        if (grid_pos not in static_obstacles and 
            grid_pos not in [mob['pos'] for mob in moving_obstacles]):
            snake_cells = [(seg[0] // CELL_SIZE, seg[1] // CELL_SIZE) for seg in snake_body]
            if grid_pos not in snake_cells:
                return [x, y]

def candidate_collision(new_head):
    x, y = new_head
    if USE_WALLS:
        if x < 0 or x >= WINDOW_WIDTH or y < 0 or y >= WINDOW_HEIGHT:
            return True
    candidate_cell = (x // CELL_SIZE, y // CELL_SIZE)
    if candidate_cell in static_obstacles:
        return True
    if USE_MOVING_OBSTACLES:
        for mob in moving_obstacles:
            if candidate_cell == mob['pos']:
                return True
    snake_body_cells = [(seg[0] // CELL_SIZE, seg[1] // CELL_SIZE) for seg in snake_body[1:]]
    if candidate_cell in snake_body_cells:
        return True
    return False

def get_state_ai():
    head_x, head_y = snake_pos
    dir_u = direction == "UP"
    dir_d = direction == "DOWN"
    dir_l = direction == "LEFT"
    dir_r = direction == "RIGHT"

    danger_straight = candidate_collision(_move_point(snake_pos, direction))
    danger_right = candidate_collision(_move_point(snake_pos, _turn_right(direction)))
    danger_left = candidate_collision(_move_point(snake_pos, _turn_left(direction)))

    if food_positions:
        apple = food_positions[0]
    else:
        apple = snake_pos
    food_left = apple[0] < head_x
    food_right = apple[0] > head_x
    food_up = apple[1] < head_y
    food_down = apple[1] > head_y

    nearest_dx = 0
    nearest_dy = 0
    dir_mob = [0, 0, 0, 0]
    if moving_obstacles:
        nearest = min(
            moving_obstacles,
            key=lambda mob: abs(mob['pos'][0]*CELL_SIZE - head_x) + abs(mob['pos'][1]*CELL_SIZE - head_y)
        )
        mob_x = nearest['pos'][0] * CELL_SIZE
        mob_y = nearest['pos'][1] * CELL_SIZE
        nearest_dx = (mob_x - head_x) / WINDOW_WIDTH
        nearest_dy = (mob_y - head_y) / WINDOW_HEIGHT
        dx, dy = nearest['dir']
        if (dx, dy) == (0, -1):
            dir_mob = [1, 0, 0, 0]
        elif (dx, dy) == (0, 1):
            dir_mob = [0, 1, 0, 0]
        elif (dx, dy) == (-1, 0):
            dir_mob = [0, 0, 1, 0]
        elif (dx, dy) == (1, 0):
            dir_mob = [0, 0, 0, 1]

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
        1 if food_down else 0,
    ]
    state.extend([nearest_dx, nearest_dy])
    state.extend(dir_mob)
    return state

def handle_collision():
    global lives, invulnerable_until
    current_time = pygame.time.get_ticks()
    if current_time < invulnerable_until:
        return
    lives -= 1
    if lives <= 0:
        game_over()
    else:
        invulnerable_until = current_time + INVULNERABILITY_TIME

def perform_boost():
    global snake_pos, last_boost_time, score
    dx, dy = direction_to_vector(direction)
    for _ in range(BOOST_DISTANCE):
        candidate = [snake_pos[0] + dx * CELL_SIZE, snake_pos[1] + dy * CELL_SIZE]
        if not USE_WALLS:
            candidate[0] %= WINDOW_WIDTH
            candidate[1] %= WINDOW_HEIGHT
        if candidate_collision(candidate):
            handle_collision()
            return True
        else:
            snake_pos[:] = candidate
            snake_body.insert(0, list(snake_pos))
            ate_apple = False
            for apple in food_positions:
                if snake_pos == apple:
                    ate_apple = True
                    score += 1
                    food_positions.remove(apple)
                    break
            if not ate_apple:
                snake_body.pop()
    last_boost_time = pygame.time.get_ticks()
    return False

def game_over(): #Highscore missing
    font = pygame.font.SysFont('consolas', 35)
    game_over_surface = font.render(f"Game Over! Score: {score}", True, RED)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (WINDOW_WIDTH / 2, WINDOW_HEIGHT / 4)
    screen.fill(DARK_GREEN)
    screen.blit(game_over_surface, game_over_rect)
    pygame.display.flip()
    pygame.time.wait(1000)
    pygame.quit()
    sys.exit()

# ----------------- Options Menu Loop -----------------
def options_menu_loop():
    menu_width = 400
    menu_height = 300
    menu_x = (WINDOW_WIDTH - menu_width) // 2
    menu_y = (WINDOW_HEIGHT - menu_height) // 2
    options_menu = OptionsMenu((menu_x, menu_y, menu_width, menu_height))
    paused = True
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            options_menu.handle_event(event)
        options_menu.draw(screen)
        pygame.display.update()
        clock.tick(60)
        if options_menu.continued:
            paused = False

# ----------------- Main Game Loop -----------------
def game_loop(model=None, device=None):
    global direction, change_to, score, level, FPS, last_boost_time, snake_pos, head_direction

    generate_static_obstacles()

    if USE_MOVING_OBSTACLES:
        for _ in range(INITIAL_MOVING_OBSTACLES_COUNT):
            spawn_moving_obstacle()

    # Create the animated boost button in the bottom-left corner.
    animated_button = RoundAnimatedButton(center=(70, WINDOW_HEIGHT - 70), radius=30,
                                          bg_color=(255, 255, 255, 100),
                                          border_color=(25, 5, 2),
                                          border_width=3,
                                          load_time=5.0)

    while True:
        while len(food_positions) < NUM_APPLES:
            food_positions.append(get_random_food_position())

        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if model is None and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    options_menu_loop()
                    continue
                if INPUT_INVERTED:
                    if event.key in (pygame.K_UP, pygame.K_w) and direction != "UP":
                        change_to = "DOWN"
                    if event.key in (pygame.K_DOWN, pygame.K_s) and direction != "DOWN":
                        change_to = "UP"
                    if event.key in (pygame.K_LEFT, pygame.K_a) and direction != "LEFT":
                        change_to = "RIGHT"
                    if event.key in (pygame.K_RIGHT, pygame.K_d) and direction != "RIGHT":
                        change_to = "LEFT"
                else:
                    if event.key in (pygame.K_UP, pygame.K_w) and direction != "DOWN":
                        change_to = "UP"
                    if event.key in (pygame.K_DOWN, pygame.K_s) and direction != "UP":
                        change_to = "DOWN"
                    if event.key in (pygame.K_LEFT, pygame.K_a) and direction != "RIGHT":
                        change_to = "LEFT"
                    if event.key in (pygame.K_RIGHT, pygame.K_d) and direction != "LEFT":
                        change_to = "RIGHT"
                if BOOST_ENABLED and event.key == pygame.K_SPACE:
                    current_time = pygame.time.get_ticks()
                    if current_time - last_boost_time >= BOOST_COOLDOWN:
                        if perform_boost():
                            break
                        continue
            animated_button.handle_event(event)

        if model is not None:
            state = get_state_ai()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if device is not None:
                state_tensor = state_tensor.to(device)
                model.to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
            change_to = action_map.get(action, change_to)

        if change_to == "UP" and direction != "DOWN":
            direction = "UP"
            head_direction = 5
        if change_to == "DOWN" and direction != "UP":
            direction = "DOWN"
            head_direction = 3
        if change_to == "LEFT" and direction != "RIGHT":
            direction = "LEFT"
            head_direction = 2
        if change_to == "RIGHT" and direction != "LEFT":
            direction = "RIGHT"
            head_direction = 0

        dx, dy = direction_to_vector(direction)
        candidate = [snake_pos[0] + dx * CELL_SIZE, snake_pos[1] + dy * CELL_SIZE]
        if USE_WALLS:
            collision = candidate_collision(candidate)
        else:
            candidate[0] %= WINDOW_WIDTH
            candidate[1] %= WINDOW_HEIGHT
            collision = candidate_collision(candidate)
        if collision:
            handle_collision()
        else:
            snake_pos[:] = candidate
            snake_body.insert(0, list(snake_pos))
            ate_apple = False
            for apple in food_positions:
                if snake_pos == apple:
                    ate_apple = True
                    score += 1
                    food_positions.remove(apple)
                    break
            if not ate_apple:
                snake_body.pop()

        update_moving_obstacles()
        if USE_MOVING_OBSTACLES:
            for mob in moving_obstacles:
                for seg in snake_body[:5]:
                    seg_cell = (seg[0] // CELL_SIZE, seg[1] // CELL_SIZE)
                    if seg_cell == mob['pos']:
                        handle_collision()
                        break

        if score >= level * 10:
            level += 1
            FPS = base_FPS + level - 1
            if USE_MOVING_OBSTACLES:
                spawn_moving_obstacle()

        draw_background()
        draw_static_obstacles()
        draw_moving_obstacles()
        for apple in food_positions:
            rect1 = pygame.Rect(apple[0], apple[1], CELL_SIZE, CELL_SIZE)
            if apple_texture:
                screen.blit(apple_texture, rect1)
            else:
                pygame.draw.rect(screen, RED, rect1)
        for index, segment in enumerate(snake_body):
            rect = pygame.Rect(segment[0], segment[1], CELL_SIZE, CELL_SIZE)
            if index == 0 and snake_head_texture:
                screen.blit(pygame.transform.rotate(snake_head_texture, 90 * head_direction), rect)
            else:
                pygame.draw.rect(screen, GREEN, rect)
        show_status(score, level, lives)

        current_time = pygame.time.get_ticks()
        animated_button.update(dt, current_time, last_boost_time, BOOST_COOLDOWN)
        animated_button.draw(screen)

        pygame.display.update()

if __name__ == "__main__":
    game_loop()
