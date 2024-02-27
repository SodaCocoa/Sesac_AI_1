import sys
from pathlib import Path

FILE = Path(__file__).resolve()
path = str(FILE.parents[1])
sys.path.append(path)

import sysInfoTitle
#sys.path.append('c:/users/hrPark/desktop/py/')
python_title_printer=sysInfoTitle.PythonTitlePrinter()
python_title_printer.sysInfo()