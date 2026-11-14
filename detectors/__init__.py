import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the registry before components to preserve registration during circular imports.
from tools.registry import DETECTOR

from detectors.audio_video.avff_detector import AVFFDetector, AVFFPretrainAdv
from detectors.audio_video.avh_detector import AVHDetector
from detectors.audio_video.c3dresnet18_seresnet_detector import C3DResNet18_SEResNet_MLP
from detectors.audio_video.fauforensics_detector import FauForensicsDetector
from detectors.audio_video.fgmdf_detector import FGMDFDetector
from detectors.audio_video.frade_detector import FRADEDetector
from detectors.audio_video.mcl_detector import MCLStage1Detector, MCLStage2Detector
from detectors.audio_video.speechforensics_detector import SpeechForensicsDetector
