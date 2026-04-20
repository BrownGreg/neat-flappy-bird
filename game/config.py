# Dimensions de la fenêtre
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 640
WIDTH = SCREEN_WIDTH
HEIGHT = SCREEN_HEIGHT
FPS = 60

# Oiseau
BIRD_X = 80
BIRD_Y = 320
BIRD_RADIUS = 18
GRAVITY = 0.5
JUMP_VELOCITY = -9.0
MAX_VELOCITY = 12.0

# Sol
GROUND_HEIGHT = 50

# Tuyaux
PIPE_WIDTH = 65

# Paliers de difficulté : (score_min, pipe_speed, gap, spawn_interval_ms)
DIFFICULTY_LEVELS = [
    (0,  2.0, 180, 1800),
    (10, 2.5, 160, 1600),
    (20, 3.0, 145, 1400),
    (35, 3.5, 130, 1300),
    (55, 4.0, 115, 1200),
]

# Couleurs
COLOR_SKY_TOP    = (30, 120, 210)
COLOR_SKY_BOTTOM = (120, 195, 245)
COLOR_GROUND     = (83, 160, 64)
COLOR_GROUND_TOP = (100, 190, 75)
COLOR_BIRD       = (255, 215, 0)
COLOR_BIRD_EYE   = (30, 30, 30)
COLOR_PIPE       = (70, 165, 55)
COLOR_PIPE_DARK  = (50, 130, 40)
COLOR_WHITE      = (255, 255, 255)
COLOR_BLACK      = (0, 0, 0)
COLOR_DARK_GREY  = (40, 40, 40)
