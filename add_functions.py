#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
from langdetect import detect
import spacy
import random
from collections import Counter
import nltk
from nltk.collocations import *
import datetime
import re
import pprint
import gensim
import gensim.corpora as corpora
from add_functions import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

import pprint
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

from sklearn.metrics import accuracy_score,f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import pickle


# In[ ]:


def parse_dates(data):
    dict_months = {"stycznia":"01","styczeń":"01",
              "lutego":"02","luty":"02",
              "marca":"03", "marzec":"03",
              "kwietnia":"04","kwiecień":"04",
              "maja":"05","maj":"05",
              "czerwca":"06","czerwiec":"06",
              "lipca":"07","lipiec":"07",
              "sierpnia":"08","sierpień":"08",
              "września":"09","wrzesień":"09",
              "października":"10","październik":"10",
              "listopada":"11","listopad":"11",
              "grudnia":"12","grudzień":"12",
              "February":"2","June":"6","May":"5","August":"8"}
    
    if data.Date[2].startswith("2020") == True:
        for i in range(len(data.Date)):
            data.Date.iloc[i] = datetime.date.fromisoformat(data.Date.iloc[i])
        return data
    elif type(re.search(r"\d\d\.",data.Date[2])) == re.Match:
        for i in range(len(data.Date)):
            try:
                data.Date.iloc[i] = datetime.datetime.strptime(data.Date.iloc[i],"%d.%m.%Y")
            except:
                data.Date.iloc[i] = datetime.datetime.strptime(data.Date.iloc[i],"%d.%m.%Y  %H:%M")
        return data
    elif "DATA" in data.Date[2]:
        for i in range(len(data.Date)):
            try:
                data.Date.iloc[i] = datetime.datetime.strptime(data.Date.iloc[i].replace("DATA: ",""),"%Y-%m-%d %H:%M:%S")
            except:
                data.Date.iloc[i] = datetime.datetime.strptime(data.Date.iloc[i].replace("DATA: ",""),"%Y-%m-%d %H:%M")
        return data
    elif type(re.search(r"\D+",data.Date[2])) == re.Match:
        for i in range(len(data.Date)):
            try:
                data.Date.iloc[i] = data.Date.iloc[i].replace(re.search(r"\D+",data.Date.iloc[i]).group().strip(),dict_months[re.search(r"\D+",data.Date.iloc[i]).group().strip()])
                data.Date.iloc[i] = datetime.datetime.strptime(data.Date.iloc[i],"%d %m %Y")
            except:
                data.Date.iloc[i] = data.Date.iloc[i]
        return data



# In[ ]:


def make_wordclouds(data,name):
    
    # reading and updating stopwords list for polish
    f = open("polish.stopwords.txt","r",encoding="utf8")
    stopwords = f.readlines()
    stopwords = [x.replace("\n","") for x in stopwords]
    adds = ["mieć","zostać","móc","pisać","chcieć","czytać"]
    stopwords = stopwords + adds
    f.close()
    stopwords = set(stopwords)
    
    content = data.processed
    lemmas_all = []
    frequencies = {}
    # reading lemmas for each text and combining them into one list
    for text in content:
        lemmas = [token.lemma_ for token in text]
        lemmas_all = lemmas_all + lemmas
    # creating frequency table for each lemma in texts
    values, counts = np.unique(lemmas_all, return_counts=True)
    for i in range(len(values)):
          frequencies[values[i]] = counts[i]
    # erasing stopwords from frequencies dictionary
    for word in stopwords:
        frequencies.pop(word,None)
    # creating word cloud from frequencies
    wordcloud = WordCloud(min_font_size=10).generate_from_frequencies(frequencies)
    
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off")
    plt.title(f"Wordcloud for {name}")
    plt.show()
    
    
    return frequencies


# In[ ]:


def find_collocations_bigram(data,key=None):

    f = open("polish.stopwords.txt","r",encoding="utf8")
    stopwords = f.readlines()
    stopwords = [x.replace("\n","") for x in stopwords]
    adds = ["mieć","zostać","móc","pisać","chcieć","czytać"]
    stopwords = stopwords + adds
    f.close()
    stopwords = set(stopwords)

    all_tokens = []
    for text in data.processed:
        tokens = [token.lemma_.lower() for token in text]
        all_tokens = all_tokens + tokens
    # finding collocations
    #reading measures
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    # finding collocations using BigramCollocationsFinder from NLTK
    finder = BigramCollocationFinder.from_words(all_tokens)
    # filtering out collocations containing stopwords
    finder.apply_word_filter(lambda w: w in stopwords)
    if key != None:
        add_filter = lambda *w: key not in w
        finder.apply_ngram_filter(add_filter)
    # applying right frequency filters based on set size
    if len(data.processed) > 500:
        finder.apply_freq_filter(50)
    elif len(data.processed) > 100 and len(data.processed)<500:
        finder.apply_freq_filter(20)
    elif len(data.processed) < 100:
        finder.apply_freq_filter(5)
    # creating a dictionary containing collocations and their frequencies
    scored = dict(finder.score_ngrams(bigram_measures.raw_freq))
    # switching to real occurence numbers
    scored.update((x, (y*len(all_tokens),y)) for x, y in scored.items())
    
    return scored

    


# In[ ]:


def find_collocations_trigram(data,key=None):
    f = open("polish.stopwords.txt","r",encoding="utf8")
    stopwords = f.readlines()
    stopwords = [x.replace("\n","") for x in stopwords]
    adds = ["mieć","zostać","móc","pisać","chcieć","czytać"]
    stopwords = stopwords + adds
    f.close()
    stopwords = set(stopwords)

    all_tokens = []
    for text in data.processed:
        tokens = [token.lemma_.lower() for token in text]
        all_tokens = all_tokens + tokens
   
    #reading measures
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    
    # finding collocations using TrigramCollocationsFinder from NLTK
    finder = TrigramCollocationFinder.from_words(all_tokens,window_size=3)
    
    # filtering out collocations containing stopwords and punctuation
    finder.apply_word_filter(lambda w: w in stopwords)
    if key != None:
        add_filter = lambda *w: key not in w
        finder.apply_ngram_filter(add_filter)
    # applying right frequency filters based on set size
    if len(data.processed) > 500:
        finder.apply_freq_filter(20)
    elif len(data.processed) < 500:
        finder.apply_freq_filter(5)
    # switching to real occurence numbers

    scored = dict(finder.score_ngrams(trigram_measures.raw_freq))
    scored.update((x, (y*len(all_tokens),y)) for x, y in scored.items())

    return scored

# In[ ]:


def clearing_texts(data):
    content = data.Content
    if "Author" in data.columns:
        authors = list(data.Author.unique())
        authors = [author.lower() for author in authors]
        for i in range(len(content)):
            for author in authors:
                content[i] = content[i].replace(author, "")


    # Removing punctuation signs from texts
    punctuation = ["/", ".", ":", ",", ")", "(", "\"", "_", "-", "?", "!", "...", "„", "”", "–", "—", "…", "[", "]",
                   "^", "'"]
    for i in range(len(content)):
        for sign in punctuation:
            content[i] = content[i].replace(sign, "")
            content[i] = content[i].replace("  ", " ")

    # dictionary with terms to clear from texts
    dict_replacements = {"czytaj także": "",
                         "czytaj też wywiad": "",
                         "wpolityce.pl": "",
                         "wpolitycepl": "",
                         "czytaj również tylko u nas": "",
                         "nasz wywiad": "",
                         "czytaj również": "",
                         "niezaleznapl": "",
                         "niezalezna.pl": "",
                         "niezależna.pl": "",
                         "niezależnapl": "",
                         "regulamin": "",
                         "portal": "",
                         "portalu": "",
                         "pch24pl": "",
                         "Copyright 2020 by STOWARZYSZENIE KULTURY CHRZEŚCIJAŃSKIEJ IM KS PIOTRA SKARGI": "",
                         "forum": "",
                         "okopress": "",
                         "oku": "",
                         "press": "",
                         "Ten artykuł nie powstałby gdyby nie wsparcie naszych darczyńców Dołącz do nich i pomóż nam publikować więcej tekstów które lubisz czytaćDziennikTematyPiszą dla nasPodcastyMultimediaNarkopolitykaO Krytyce PolitycznejKontaktWspieraj nas© 2020 Krytyka Polityczna Wszystkie prawa zastrzeżone | Partner technologiczny MeverywhereplSzukajKrajŚwiatKulturaGospodarkaPiekło kobietKoronawirusWybory w USAKlimatPracaWspieraj nas KsięgarniaO nasO Krytyce PolitycznejJesteśmy stowarzyszeniemKontaktAbout usŚwietlica w GdańskuŚwietlica w CieszynieJasna 10 Świetlica w WarszawieInstytutBaza ekspertek Akceptuję regulamin i politykę prywatności'": "",
                         "krytyka": "",
                         "krytyce": "",
                         "polityczna": "",
                         "politycznej": "",
                         "dziękujemyklub inteligencji katolickiejfreta 2024a00227 warszawapolityka prywatności ***redakcja@\u200bkikwawplfacebookcommagazynkontaktplpoglądy wyrażane przez autorkii autorów tekstów nie są tożsamez poglądami wydawcymagazyn kontakt korzysta z dofinansowania pochodzącego z niwcrso w ramach proo3 na lata 20182030 dofinansowano ze środkówministra kulturyi dziedzictwa narodowego": "",
                         "gazetapl": "",
                         "gazeta": "",
                         "fot": "",
                         "Rozwijamy nasz serwis dzięki wyświetlaniu reklamBlokując reklamy nie pozwalasz nam tworzyć wartościowych treści Wyłącz AdBlock i odśwież stronę Żaden utwór zamieszczony w serwisie nie może być powielany i rozpowszechniany lub dalej rozpowszechniany w jakikolwiek sposób w tym także elektroniczny lub mechaniczny na jakimkolwiek polu eksploatacji w jakiejkolwiek formie włącznie z umieszczaniem w Internecie bez pisemnej zgody właściciela praw Jakiekolwiek użycie lub wykorzystanie utworów w całości lub w części z naruszeniem prawa tzn bez właściwej zgody jest zabronione pod groźbą kary i może być ścigane prawnie": "",
                         "Żaden utwór zamieszczony w serwisie nie może być powielany i rozpowszechniany lub dalej rozpowszechniany w jakikolwiek sposób w tym także elektroniczny lub mechaniczny na jakimkolwiek polu eksploatacji w jakiejkolwiek formie włącznie z umieszczaniem w Internecie bez pisemnej zgody właściciela praw Jakiekolwiek użycie lub wykorzystanie utworów w całości lub w części z naruszeniem prawa tzn bez właściwej zgody jest zabronione pod groźbą kary i może być ścigane prawnie": "",
                         "Dofinansowano ze środków Ministra Kultury i Dziedzictwa Narodowego pochodzących z Funduszu Promocji Kultury Ten utwór z wyłączeniem grafik jest udostępniony na licencji Creative Commons Uznanie Autorstwa 40 Międzynarodowe Zachęcamy do jego przedruku i wykorzystania Prosimy jednak o zachowanie informacji o finansowaniu artykułu oraz podanie linku do naszej strony": "",
                         "super express": "",
                         "klub jagiellonski": ""
                         }
    for i in range(len(content)):
        for word, initial in dict_replacements.items():
            content[i] = content[i].lower().replace(word.lower(), initial)

    for i in range(len(content)):
        if "**" in content[i]:
            content_split = content[i].split("**")
            content[i] = content[0]

    data.Content = content

    return data

# In[ ]:


def find_best_model(X_train,X_test,y_train,y_test,pipelines,param_grids):
    # using grid search cv to find the best model

    # defining variables to store parameters and scores for the best model
    best_model = ""
    best_params = ""
    score_f1 = 0
    acc_score = 0
    
    # testing in a loop for each pipeline - parameters combination
    for i in range(len(pipelines)):
        
        # Creating GridSearch object
        grid = GridSearchCV(pipelines[i],
                            param_grid=param_grids[i],
                            refit=True,      # refitting best model
                            cv = 5,          # nr of crossvalidations
                            verbose=3,       # provides information during the learning process
                            n_jobs = 3)      # nr of cores used for testing
        
        # Fitting data
        grid.fit(X_train,y_train)
        
        # Predicting on test data, scoring
        y_pred = grid.predict(X_test)
        test_f1 = f1_score(y_test,y_pred)
        test_acc = accuracy_score(y_test,y_pred)
        
        # Checking if model is better than current best, declaring new values if it is
        if test_f1 > score_f1:
            acc_score = test_acc
            score_f1 = test_f1
            best_model = pipelines[i]
            best_params = grid.best_params_
            
    
    # printing best model 
    print(f"Best model: {best_model} with {best_params} parameters. F1 score = {score_f1} and Accuracy score = {acc_score}")
    return best_model
