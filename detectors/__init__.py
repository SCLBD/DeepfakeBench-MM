import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.registry import DETECTOR

from detectors.our_baseline import BaselineStage2Detector
from detectors.mds_detector import MDSDetector
from detectors.avts_detector import AVTSStage2Detector