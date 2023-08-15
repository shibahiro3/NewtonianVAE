import os
import sys

sys.path.append("src")

# from dm_control import suite
from third.dm_control import suite


def main():
    # Iterate over a task set:
    for domain_name, task_name in suite.BENCHMARKING:
        print(f'domain_name="{domain_name}", task_name="{task_name}"')

    # Load one task :
    # env = suite.load(domain_name, task_name)


if __name__ == "__main__":
    main()
