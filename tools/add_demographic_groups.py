"""Compatibility entry point for demographic manifest annotation."""

import sys

from pathlib import Path


project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from commands.fairness.add_demographic_groups import (
    annotate_demographic_groups,
    group_from_path,
    main,
    parse_args,
    update_manifest,
)


if __name__ == '__main__':
    main()
