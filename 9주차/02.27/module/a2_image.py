from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('My First Application')
        self.move(300, 0)
        
        label2 = QLabel(self)
        pixmap = QPixmap(r'C:\Users\bluecom011\Desktop\Sesac_AI\9주차\02.27\module\pic1.png')
        
        # 이미지 크기 조정
        scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        label2.setPixmap(scaled_pixmap)
        
        # 창 크기를 조정된 이미지 크기에 맞춤
        self.resize(scaled_pixmap.width(), scaled_pixmap.height())
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
