from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
 

SOURCES_LIST = [IMAGE, VIDEO]

# Images config
IMAGES_DIR = '/home/marroula/aziz_cleeve/CRMN_INTERFACE/images'
DEFAULT_IMAGE = '/home/marroula/aziz_cleeve/CRMN_INTERFACE/images/office_4.jpg'
DEFAULT_DETECT_IMAGE = '/home/marroula/aziz_cleeve/CRMN_INTERFACE/images/office_4_detected.jpg'

# Videos config
VIDEO_DIR = '/home/marroula/aziz_cleeve/CRMN_INTERFACE/videos'
VIDEOS_DICT = {
    'video_0': '/home/marroula/aziz_cleeve/CRMN_INTERFACE/videos/video.MOV',
    'video_1': '/home/marroula/aziz_cleeve/CRMN_INTERFACE/videos/tortues.mp4',
    'video_2': '/home/marroula/aziz_cleeve/CRMN_INTERFACE/videos/tortuess.mp4',
    'video_3': '/home/marroula/aziz_cleeve/CRMN_INTERFACE/videos/videoplayback.mp4',
    'video_4': '/home/marroula/aziz_cleeve/CRMN_INTERFACE/videos/plongeur.mp4',
}

# ML Model config
MODEL_DIR = '/home/marroula/aziz_cleeve/CRMN_INTERFACE/weights'
#DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
DETECTION_MODEL_YOLOv9 = '/home/marroula/aziz_cleeve/CRMN_INTERFACE/weights/bestV9KAGGLE.pt'
DETECTION_MODEL_YOLOv8 = '/home/marroula/aziz_cleeve/CRMN_INTERFACE/weights/best.pt'
DETECTION_MODEL_FASTER = '/home/marroula/aziz_cleeve/CRMN_INTERFACE/weights/FRCNN-V0-5epochs.pth'
#SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
