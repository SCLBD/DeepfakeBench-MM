import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.registry import LOSSFUNC
from losses.standard_loss import CrossEntropyLoss
from losses.mds_loss import MDSL1
# from losses.xxx import xxx