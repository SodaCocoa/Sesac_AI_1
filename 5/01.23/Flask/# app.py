# app.py
from flask import Flask, render_template

# Flask 객체 인스턴스 생성
app = Flask(__name__)

import pandas as pd
datas=pd.read_csv(r'C:\data\Flask\data.csv')
datas=datas.to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index.html', datas=datas)

@app.route('/table')
def index_table():
    return render_template('index_table.html', datas=datas)

if __name__ == '__main__':
    app.run(debug=True)
