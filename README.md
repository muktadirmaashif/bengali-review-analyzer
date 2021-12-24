# Bengali Review Analyzer - A Flask app for analyzing the sentiment of bengali reviews from online Bookshop

Welcome to bengali-review-analyzer. It is a machine learning based app to analyze bengali book reviews. For a given url, it will scrape all the reviews from the website and analyze the sentiment. Final output will be the percentage of positive and negative reviews along with the book details (cover image, book title and author name).

## Installation Procedure

1. Clone the git repository
2. Create a python virtual environment
```shell
$ python -m venv <venv-name>
```

3. cd to the venv and activate it
```shell
$ cd <venv-name>
$ source bin/activate
```
4. Install all the packages from requirements.txt
```shell
$ pip install -r requirements.txt
```
5. Flask would need to detect the main app file. In this case, it is main.py file. Also FLASK_ENV needs to be set as development to enable the debug mode.
```shell
$ export FLASK_APP=main
$ export FLASK_ENV=development
$ flask run
```
***
Don't worry if it takes a little time. there's a machine learning model is being trained in the back-end. Therefore it will take some time to load first time. 
***
## SABR Homepage
![SABR Homepage](/static/images/bengali-review-analyzer.png)

***
## Live demo [HERE](https://bengali-review-analyzer.herokuapp.com). 
Thanks!

***
## Credit 
1. [https://github.com/rukhat/Rokomari-Review-Scraper](https://github.com/rukhat/Rokomari-Review-Scraper)
2. [https://github.com/eftekhar-hossain/Bengali-Restaurant-Reviews](https://github.com/eftekhar-hossain/Bengali-Restaurant-Reviews)
