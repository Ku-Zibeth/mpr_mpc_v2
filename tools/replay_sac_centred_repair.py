"""Historical command alias; online stopping certification is retired."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('smoke_mppi_v3.py')), run_name='__main__')
