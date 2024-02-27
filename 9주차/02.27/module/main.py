import test
from pathlib import Path
#상위경로 잡기
FILE = Path(__file__).resolve()
path = str(FILE.parents[1])+'\\txt\\'

#ROOT= FILE.parents[0]
#ROOT1= FILE.parents[1]
#print(FILE)
#print(ROOT)
#print(ROOT1)

file = 'val.txt'

with open(path+file, 'r') as f:
    content = f.readlines()

test.prn(content)
