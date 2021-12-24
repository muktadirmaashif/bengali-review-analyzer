from flask import Flask, render_template, request, url_for, redirect, flash
#from kivy.uix.screenmanager import Screen
import Scraper 
from Scraper import *
import validators

app = Flask(__name__)
app.secret_key = 'super secret'

""" @app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
 """
@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        is_url = validators.url(url)
        if (is_url):
            
            print(url)
            main(url)
            
            scrap = {'title': Scraper.Bangla_title,
                        'author': Scraper.Bangla_author,
                        'pos': mlModel.posPercentage,
                        'neg': mlModel.negPercentage}
            
            
            return render_template("details.html", scrap= scrap)
        else:
            flash('Invalid URL! Please enter a valid book URL from rokomari.com.', 'error')
            return render_template('index.html')

    else:
        return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')
    
 

if __name__ == "__main__":
    app.run(debug=True)