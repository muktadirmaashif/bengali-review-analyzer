import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns 
import re, json, nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from utils import cleaned_reviews, stopwords_info, stopword_removal, process_reviews, performance_table, calc_unigram_tfidf
from utils import calc_unigram_tfidf,calc_bigram_tfidf,calc_trigram_tfidf,show_tfidf, label_encoding,dataset_split, model_performace, ml_models_for_trigram_tfidf

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

stopwords_list = 'stopwords-bn.txt'

#importing the dataset

data = pd.read_excel('BookReviews.xlsx')#bookreview1.xlsx
""" print("Total Reviews:",len(data),
      "\nTotal Positive Reviews:",len(data[data.Sentiment =='positive ']),
      "\nTotal Negative Reviews:",len(data[data.Sentiment=='negative']))
 """
# Apply the function into the dataframe
data['cleaned'] = data['Reviews'].apply(process_reviews,stopwords = stopwords_list,removing_stopwords = True)  

# print some cleaned reviews from the dataset
sample_data = [11,1000]
#for i in sample_data:
      #print('Original:\n',data.Reviews[i],'\nCleaned:\n',data.cleaned[i],'\n','Sentiment:-- ',data.Sentiment[i],'\n')   

# Length of each Reveiws
data['length'] = data['cleaned'].apply(lambda x:len(x.split()))
# Remove the reviews with least words
dataset = data.loc[data.length>1]
dataset = dataset.reset_index(drop = True)
""" print("After Cleaning:","Removed {} Small Reviews".format(len(data)-len(dataset)),
      "\nTotal Reviews:",len(dataset),
      "\nTotal Positive Reviews:",len(dataset[dataset.Sentiment =='positive ']),
      "\nTotal Negative Reviews:",len(dataset[dataset.Sentiment=='negative']),
     "\nTotal Neutral Reviews:",len(dataset[dataset.Sentiment=='neutral']))
 """
#save the cleaned data 
dataset[['cleaned','Sentiment']].to_excel('clean_rr_reviews.xlsx')

# open a file, where you want to store the data
file = open('rr_review_data.pkl', 'wb')
# dump information to that file
pickle.dump(data, file)

# load the save file
data = open('rr_review_data.pkl','rb')
data = pickle.load(data)

# Stopwords pickle 
stp = open(stopwords_list,'r', encoding='utf-8').read().split()
# open a file, where you ant to store the data
file = open('rr_stopwords.pkl', 'wb')
# dump information to that file
pickle.dump(stp, file)

stp = open('rr_stopwords.pkl','rb')
stp = pickle.load(stp)
print(len(stp))

# calculate the Tri-gram Tf-idf feature
cv,feature_vector = calc_trigram_tfidf(dataset.cleaned) 
# Encode the labels
lables = label_encoding(dataset.Sentiment,False)
# Split the Feature into train and test set
X_train,X_test,y_train,y_test = dataset_split(feature_space=feature_vector,sentiment=lables)

# Classifiers Defination
ml_models,model_names = ml_models_for_trigram_tfidf()             

# call model accuracy function and save the metrices into a dictionary
accuracy = {f'{model_names[i]}':model_performace(model,X_train,X_test,y_train,y_test) for i,model in enumerate(ml_models)}
# Save the performance parameter into json file
with open('ml_performance_trigram.json', 'w') as f:
    json.dump(accuracy, f)

# Load the json file
accuracy = json.load(open('ml_performance_trigram.json'))
table = performance_table(accuracy)
#print(table)


#Final Model
# calculate the Tri-gram Tf-idf feature
cv,feature_vector = calc_trigram_tfidf(dataset.cleaned) 
# Encode the labels
lables = label_encoding(dataset.Sentiment,False)
# Split the Feature into train and test set
X_train,X_test,y_train,y_test = dataset_split(feature_space=feature_vector,sentiment=lables)


sgd_model = SGDClassifier(loss ='log',penalty='l2', max_iter=5)
sgd_model.fit(X_train,y_train) 
y_pred = sgd_model.predict(X_test)
accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)*100
#print(accuracy)



# open a file, where you ant to store the data
file = open('rr_review_sgd.pkl', 'wb')

# dump information to that file
pickle.dump(sgd_model, file)
pickle.dump(sgd_model, file)
model = open('rr_review_sgd.pkl','rb')
sgd = pickle.load(model)
#y_pred = sgd.predict(X_test)

#Check a review sentiment using our model

data1 = pd.read_csv('Reviews.csv')
for i in data1.Review:
    l= data1['Review'].apply(process_reviews,stopwords = stopwords_list,removing_stopwords = True)
       
#print(l)
def analyzeReview(l):
    neg = 0;pos= 0;
    for i in l:   
        if (len(i))>0:
            # calculate the Unigram Tf-idf feature
            cv,feature_vector = calc_trigram_tfidf(dataset.cleaned) 
            feature = cv.transform([i]).toarray()

            sentiment = sgd.predict(feature)
            score = round(max(sgd.predict_proba(feature).reshape(-1)),2)*100
            #print(i)
            if (sentiment ==0):
                #print(f"It is a Negative Review and the probability is {score}%")
                neg=neg+1
            else:
                #print(f"It is a Positive Review and the probability is {score}%")
                pos=pos + 1

    totalRev = len(l)
    #print("Total Reviews : ",len(l))
    cleaned = totalRev - (neg+pos)
    #print("Cleaned review : ", cleaned)
    #print("Total negative reviews : ",neg)
    #print("Total positive reviews : ",pos)
    global negPercentage
    global posPercentage


    negPercentage = neg/(totalRev-cleaned)*100
    negPercentage = round(negPercentage, 2)
    #print(f'Negative review percentage: {negPercentage}%') 
    
    posPercentage = pos/(totalRev-cleaned)*100
    posPercentage = round(posPercentage, 2)
    #print(f'Positive review percentage: {posPercentage}%') 
    