from ultralytics import YOLO

##################################################################################################
# CONFIGURATION
##################################################################################################

# Paths (hardcoded for now)
MODEL_PATH = "C:/cvr/funscript-generator/VR-Funscript-AI-Generator/models/k00gar-11n-200ep-best.pt"


##################################################################################################
# DETECTION
##################################################################################################

# VIDEO PREPARATION
RENDER_RESOLUTION = 1080
TEXTURE_RESOLUTION = 1440
PITCH=-10
YAW=0

# YOLO
YOLO_CONF = 0.3
YOLO_BATCH_SIZE = 8
# TODO load the model conditionally
YOLO_MODEL = YOLO(MODEL_PATH, task="detect")


##################################################################################################
# ADVANCED / DEVELOPMENT
##################################################################################################

UPDATE_PROGRESS_INTERVAL = 0.25 # Updates progress in the console and in gui
# when enabled the queue will be processed one by one (use it on (QUEUE_MAXSIZE / frame rate) seconds longer videos or less)
# raw frames take a lot of memory (RAM) so don't set the queue to high
SEQUENTIAL_MODE = False
QUEUE_MAXSIZE = 3000 if SEQUENTIAL_MODE else 300 # Bounded queue size to avoid memory blow-up
DEBUG_PATH = "C:/cvr/funscript-generator/tmp_output"
PROGRESS_BAR = True # disable when you want to print messages while debugging
