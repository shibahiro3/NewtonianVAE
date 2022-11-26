"""
Ref:
    https://github.com/deepmind/dm_control/blob/main/dm_control/suite/reacher.py
    https://github.com/deepmind/dm_control/blob/main/dm_control/suite/reacher.xml

    https://github.com/deepmind/dm_control/blob/main/dm_control/suite/point_mass.py
    https://github.com/deepmind/dm_control/blob/main/dm_control/suite/point_mass.xml

    ...
"""

import sys
from pathlib import Path
from pprint import pprint

site_packages = []
for p in sys.path:
    if Path(p).name == "site-packages":
        site_packages.append(p)

domain = sys.argv[1]

if len(site_packages) > 0:
    for sp in site_packages:
        suite = Path(sp, "dm_control", "suite")
        if suite.exists():
            pre = sys.argv[2] if len(sys.argv) > 2 else ""
            Path(suite, f"{domain}.py").write_text(Path(pre, f"{domain}.py").open().read())
            Path(suite, f"{domain}.xml").write_text(Path(pre, f"{domain}.xml").open().read())

            print("write to:")
            print(Path(suite, f"{domain}.py"))
            print(Path(suite, f"{domain}.xml"))
            break
else:
    print("Sorry...")
