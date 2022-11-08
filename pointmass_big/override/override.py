import sys
from pathlib import Path
from pprint import pprint

site_packages = []
for p in sys.path:
    if Path(p).name == "site-packages":
        site_packages.append(p)

domain = "point_mass"

if len(site_packages) > 0:
    for sp in site_packages:
        suite = Path(sp, "dm_control", "suite")
        if suite.exists():
            Path(suite, f"{domain}.py").write_text(Path(f"{domain}.py").open().read())
            Path(suite, f"{domain}.xml").write_text(Path(f"{domain}.xml").open().read())
            print("Done")

            print("write to:")
            print(Path(suite, f"{domain}.py"))
            print(Path(suite, f"{domain}.xml"))
else:
    print("Sorry...")
