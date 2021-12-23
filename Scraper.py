from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import requests
from utils import process_reviews
import mlModel
stopwords_list = 'stopwords-bn.txt'


def main(url):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.minimize_window()
    
    # url = 'https://www.rokomari.com/book/8450/'
    #url = 'https://www.rokomari.com/book/206592/'
    
    driver.get(url)

    titles = driver.title.replace(' | Rokomari.com', '').split(' - ')
    
    global Bangla_title
    Bangla_title = titles[0].split(': ')[0]
    global Bangla_author
    Bangla_author = titles[0].split(': ')[1]
    
    English_title = titles[1].split(': ')[0]
    English_author = titles[0].split(': ')[1]

    namesRaw = driver.find_elements_by_class_name('name')
    reviewsRaw = driver.find_elements_by_class_name('review-text')

    names = []
    reviews = []

    for n in namesRaw:
        if n.text != '':
            names.append(n.text.replace('By ', '').replace(',', ''))

    for r in reviewsRaw:
        if r.text != '':
            reviews.append(r.text)

    # Downlaod Cover Image
    cover_image = driver.find_elements_by_class_name('look-inside')
    image_src = cover_image[0].get_attribute('src')
    img = requests.get(image_src)

    image_file = open('static/images/CoverImage.jpg', 'wb')
    for chunk in img.iter_content(100000):
        image_file.write(chunk)
    image_file.close()
    
    
    
    data = {'Name': names, 'Review': reviews}
    

    frame = pd.DataFrame(data)
    file_name = 'Reviews.csv'
    frame.to_csv(file_name, header=True, index=False, encoding='utf-8')

    driver.quit()



    data1 = pd.read_csv('Reviews.csv')
    for i in data1.Review:
        l= data1['Review'].apply(process_reviews,stopwords = stopwords_list,removing_stopwords = True)

    mlModel.analyzeReview(l)
    
    
if __name__ == '__main__':
    main()