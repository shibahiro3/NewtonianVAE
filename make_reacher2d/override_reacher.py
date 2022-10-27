"""
Ref:
    https://github.com/deepmind/dm_control/blob/main/dm_control/suite/reacher.py
    https://github.com/deepmind/dm_control/blob/main/dm_control/suite/reacher.xml
"""

import sys
from pathlib import Path
from pprint import pprint

site_packages = None
for p in sys.path:
    if Path(p).name == "site-packages":
        site_packages = p

if site_packages is not None:
    Path(site_packages, "dm_control", "suite", "reacher.py").write_text(
        Path("reacher.py").open().read()
    )
    Path(site_packages, "dm_control", "suite", "reacher.xml").write_text(
        Path("reacher.xml").open().read()
    )
else:
    print("Sorry...")
