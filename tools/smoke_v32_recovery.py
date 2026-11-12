"""Historical CLI alias; Recovery no longer exists in the active planner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.smoke_mppi_v3 import main

if __name__ == '__main__':
    main()
