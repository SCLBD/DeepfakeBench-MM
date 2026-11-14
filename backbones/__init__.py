import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the registry before components to preserve registration during circular imports.
from tools.registry import BACKBONE

from backbones.audio_backbones.audiomae_backbone import AudioMAEBackbone
from backbones.video_backbones.marlin_backbone import MARLINBackbone
