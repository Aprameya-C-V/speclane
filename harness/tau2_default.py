"""The default tau2 agent, run through the same command line plus the shared judge patch."""
import sys
import tau2_judge  # noqa: F401
from tau2.cli import main

if __name__ == "__main__":
    sys.argv[0] = "tau2"
    main()
