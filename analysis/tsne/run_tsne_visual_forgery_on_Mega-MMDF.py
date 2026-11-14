"""Draw eight visual forgery categories, combining visual manipulation and FR."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.tsne.forgery_groups import main

if __name__ == '__main__':
    main('visual')
