from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def home():
    if request.method == 'GET':
        return render_template('stock.html')

    if request.method == 'POST':
        stockName = (request.form['stockName'])

        url = 'http://companyinfo.stock.naver.com/v1/company/c1010001.aspx?cmp_cd='+ stockName
        datas = pd.read_html(url, encoding='utf-8')[12]
        datas = datas.to_dict(orient="records")
        datainfo=pd.read_html(url,encoding='utf-8')[0]
        datainfo=datainfo.iloc[:,0][0]
        return render_template('stock.html', datas=datas, datainfo=datainfo)



if __name__ == '__main__':
    app.run(debug=True)
