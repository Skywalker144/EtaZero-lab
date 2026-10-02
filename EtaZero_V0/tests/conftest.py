from pathlib import Path
import os
import sys

python_root=str(Path(__file__).resolve().parents[1]/"python")
sys.path.insert(0,python_root)
# Recovery tests spawn fresh interpreters, which do not inherit sys.path.
python_path=os.environ.get('PYTHONPATH')
os.environ['PYTHONPATH']=python_root+(os.pathsep+python_path if python_path else '')
