from flask import Flask, render_template
app = Flask(__name__)
# 이미지 파일명 리스트 (static 폴더 내의 상대 경로로 지정)
image_files = [
    "images_1.jpg",
    "images_2.jpg",
    "images_3.jpg",
    "images_4.jpg",
    "images_5.jpg",
    "images_6.jpg"
    # 추가 이미지 파일들
]
@app.route('/')
def index():
    return render_template('index2.html', image_files=image_files)
if __name__ == '__main__':
    app.run(debug=True)