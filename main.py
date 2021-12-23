from flask import Flask, render_template, request, url_for, redirect
from kivy.uix.screenmanager import Screen
import Scraper 
from Scraper import *


app = Flask(__name__)

""" @app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
 """
@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        print(url)
        main(url)
        
        scrap = {'title': Scraper.Bangla_title,
                    'author': Scraper.Bangla_author,
                    'pos': mlModel.posPercentage, 
                    'neg': mlModel.negPercentage}
        return render_template("details.html", scrap= scrap)
    else:
        return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')
    
 

if __name__ == "__main__":
    app.run(debug=True)