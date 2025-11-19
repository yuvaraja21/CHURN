
import sys, pprint
from pathlib import Path
print("cwd:", Path.cwd())
pprint.pp(sys.path[:6])
try:
    import src.train
    print("SUCCESS: imported src.train")
except Exception as e:
    print("IMPORT FAILED:", type(e).__name__, e)