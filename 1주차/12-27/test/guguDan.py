''''
구구단 출력 프로그램
input 명령에 의해 생성되는 데이터 입력란에 숫자를 입력함
input 함수는 문자열로 자료를 받음으로 int값으로 데이터 타입 변환함
'''


def gugu():
    x=input('출력할 구구단의 단을 선택해주세요. 1~9')
    x=int(x)
    for num in range(9):
        print('%d*%d=%d'%(x,num,x*num))