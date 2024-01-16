# C:/doit/game/__init__.py
from .graphic.render import render_test

VERSION = 3.5

def print_version_info():
    print(f"The version of this game is {VERSION}.")

# 여기에 패키지 초기화 코드를 작성한다.
print("Initializing game ...")
#이렇게 하면 패키지를 처음 import할 때 초기화 코드가 실행된다.
#단, 초기화 코드는 한 번 실행된 후에는 다시 import를 수행하더라도 실행되지 않는다. 
#예를 들어 다음과 같이 game 패키지를 import한 후에 하위 모듈을 다시 import 하더라도 초기화 코드는 처음 한 번만 실행된다.

