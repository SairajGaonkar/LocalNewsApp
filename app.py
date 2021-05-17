import atexit
from functools import cmp_to_key
import FetchZipCodeDetails
from flask import request
import requests
import xlsxwriter
from flask import Flask
import nltk
from newspaper import Article
from newspaper import Config
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import pymongo
from pymongo import MongoClient
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import string
import numpy as np
import warnings
##
from flask import Blueprint, request, Response, jsonify
import json
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import jwt
import datetime
from functools import wraps
import uuid
import MongoDbConnection
import sys
from fetchTweets import *
from apscheduler.schedulers.background import BackgroundScheduler
from bson.json_util import dumps, loads
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import math

warnings.filterwarnings('ignore')

app = Flask(__name__)

SECRET_KEY = "KMLDJFW90988F8392JFFAFDGKJ"

'''MongoDB connections'''
db = MongoDbConnection.connectMongo()
Users_collections = db['users']


# twitter_collections = db['twitter']
# db = database['news']


@app.route("/zipcode")
def get_details_from_zip():
    details = FetchZipCodeDetails.getZipCodeDetails(request.args.get('zip'))
    print(details[0])

    return details[0]


@app.route("/news")
def home():
    # Zip Code Integration
    details = FetchZipCodeDetails.getZipCodeDetails(request.args.get('zip'))
    if details:
        country = details[0]['country']
        geo = details[0]['city'] + ', ' + details[0]['state']
    else:
        return {
            "error": "Zipcode invalid"
        }
    # Set up configs for newspaper3k, mongoDB, and RapidAPI

    # RapidAPI
    url = "https://google-news.p.rapidapi.com/v1/geo_headlines"
    headers = {
        'x-rapidapi-key': "904b6bd2e1msh3bc1c964245efc5p197415jsn5c86a2e70272",
        'x-rapidapi-host': "google-news.p.rapidapi.com"
    }
    querystring = {"lang": "en", "country": country, "geo": geo}

    # Newspaper3k
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent

    # cluster = MongoClient(
    #     "mongodb://LocalNews:LNews4SoCoAtLuddy@socolab.luddy.indiana.edu/LocalNews?tls=true")
    # db = cluster["news"]

    top = fetch_top_40_words(querystring, db)
    if (len(top) > 0):
        print("City already present, using existing data.")
        # print(top)
        # return "Done"
        json_Top_Words = json.dumps(top)
        return json_Top_Words

    # Get today's news
    print("Getting News..")
    all_news = get_news(querystring, url, headers)
    if (len(all_news) > 2):
        print("News Fetch - Done")
    else:
        print("API Request Timeout")
        return

    # Make corpus and dictionary to upload to mongoDB
    corpus, d = make_corpus(all_news, config)
    print("Make Corpus - Done")

    # Note: this will modify d and add a new key - keywords
    # keywords_all has all the keywords of individial article with tf-idf score
    keywords_all = compute_top_words(corpus, d)
    if (keywords_all is None):
        print("Failed to compute keywords")
    else:
        print("Keywords compute - Done")

    # Compute keywords by k-means
    keywords_all = kmeans_top_words(corpus)
    if(keywords_all is None):
        print("Failed to compute keywords")
    else:
        print("K-Means keywords compute, done")
    

    # Push the articles along with keywords into database
    # try:
    #     push_into_db(d, querystring, db)
    # except:
    #     print("Couldn't push to database -- 1")

    # Get top n words for database along with their most relevant article link
    data = get_top_n_words_precomputed(keywords_all, 40)
    articles, links, summary = get_link_to_article_all(data, d)
    data = make_dataframe_top_words(data, articles, links, summary)
    data = data.reset_index()
    data.columns = ['words', 'tfidf', 'link', 'title', 'summary']
    print("Get top n words - Done")

    top_words = change_dataframe_to_list(data)
    # Push top n words into database
    try:
        push_top_word_into_db(top_words, querystring, db)
    except:
        print("Couldn't push into database -- 2")

    delete_old(querystring, db)

    top = fetch_top_40_words(querystring, db)
    # return "Done"
    # print(top)
    json_Top_Words = json.dumps(top)
    return json_Top_Words


def fetch_top_40_words(querystring, db):
    # Collection Name
    col = querystring['geo']
    now = datetime.datetime.now()
    todays_date = now.strftime("%m_%d_%Y")
    top_words = db[col].find_one({"date": todays_date})
    top = {}
    if top_words:
        # for doc in top_words:
        top['date'] = top_words['date']
        top['data'] = top_words['data']
        # top['time'] = top_words['time']

    return top

# Get old news


@app.route("/oldnews")
def fetch_old_top_40_words():
    details = FetchZipCodeDetails.getZipCodeDetails(request.args.get('zip'))
    index = int(request.args.get('index'))
    print(index)

    if details:
        country = details[0]['country']
        geo = details[0]['city'] + ', ' + details[0]['state']
    else:
        return {
            "error": "Zipcode invalid",
            "code": 501
        }
    querystring = {"lang": "en", "country": country, "geo": geo}
    col = querystring["geo"]
    past_days = 3
    docs = db[col].find({})
    list_docs = []
    for doc in docs:
        list_docs.append(doc)

    # list_docs = sorted(list_docs, key=lambda x: datetime.datetime.strptime(x['_id'], "%Y_%m_%d_%H_%M_%S"), reverse=True)
    list_docs = sorted(list_docs, key=lambda x: datetime.datetime.strptime(
        x['date'], "%m_%d_%Y"), reverse=True)
    list_docs = list_docs[:past_days]
    print(list_docs[0]['date'])
    #index = 1
    try:
        top = {}
        if list_docs:
            # for doc in top_words:
            top['date'] = list_docs[index]['date']
            top['data'] = list_docs[index]['data']
            # top['time'] = top_words['time']
        json_Top_Words = json.dumps(top)
        return json_Top_Words
    except:
        return {
            'error': "Error in fetching old news",
            'code': 500
        }


# Get old tweets
@app.route("/oldtweets", methods=['Post'])
def fetch_old_tweets():
    details = FetchZipCodeDetails.getZipCodeDetails(request.args.get('zip'))
    index = int(request.args.get('index'))
    print(index)
    past_days = 3
    geo = ''
    if details:
        geo = details[0]['city'] + ', ' + details[0]['state']
    else:
        return {
            "error": "Zipcode invalid",
            "code": 501
        }
    geo = geo + '_twitter'
    print(geo)
    docs = db[geo].find({})
    print("docs", docs)
    list_docs = []

    for doc in docs:
        list_docs.append(doc)

    print(f"Number of entries right now: {len(list_docs)}")
    # list_docs = sorted(list_docs, key=lambda x: datetime.datetime.strptime(x['_id'], "%Y_%m_%d_%H_%M_%S"), reverse=True)
    list_docs = sorted(list_docs, key=lambda x: datetime.datetime.strptime(
        x['date'], "%b-%d-%Y"), reverse=True)
    list_docs = list_docs[:past_days]
    try:
        json_top_tweets = json.dumps(list_docs[index])
        return json_top_tweets
    except:
        return {
            'error': "Error in fetching old tweets",
            'code': 500
        }


# Get latest news from api


def get_news(querystring, url, headers):
    response = requests.request(
        "GET", url, headers=headers, params=querystring)

    all_news = response.json().get("articles", [])
    return all_news


###################
## Preprocessing ##
###################

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def lemmatization(words):
    lem = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    new_words = []
    for word in words:
        if not word in stop and len(word) != 1:
            try:
                word = lem.lemmatize(word)
            except:
                e = sys.exc_info()[0]
            new_words.append(word)

    return new_words


def stemming(words):
    ps = PorterStemmer()
    stop = set(stopwords.words('english'))
    words = [ps.stem(word)
             for word in words if not word in stop and len(word) != 1]
    return words


def remove_words(words):
    #     remove_digits = str.maketrans('', '', digits)
    #     for i in range(words):
    #         words[i] = s.translate(words[i])
    for i in range(len(words)):
        words[i] = ''.join([i for i in words[i] if not i.isdigit()])
    words = [word for word in words if len(word) > 2]
    return words


def preprocessing(text):
    text = remove_URL(text)
    text = remove_html(text)
    text = remove_punct(text)

    # Tokenize
    words = word_tokenize(text)
    # Remove digits / numbers
    words = remove_words(words)
    # Lemmatization
    words = lemmatization(words)

    # Stemming Skipped for now will ask Professor
    #     words = stemming(words)

    text = ' '.join(words)
    text = text.lower()
    return text


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


# Make corpus for all the news
def make_corpus(news, config):
    d = {}
    corpus = {}
    item = 0

    printProgressBar(0, len(news), prefix='Progress:',
                     suffix='Complete', length=50)
    for every_news in news:
        id = every_news.get("id")
        title = every_news.get("title")
        link_url = every_news.get("link")

        page = Article(link_url.strip(' \" '), config=config)
        if id not in d:
            d[id] = {}
            d[id]['title'] = title
            d[id]['link'] = link_url
            try:
                page.download()
                page.parse()
                page.nlp()
                summary = page.summary
                description = page.text
                d[id]['summary'] = summary
                d[id]['description'] = description
                # Change summary -> description
                summary = preprocessing(description)
                title = preprocessing(title)
                corpus[id] = (title + summary)
                item += 1
                printProgressBar(
                    item, len(news), prefix='Progress:', suffix='Complete', length=50)
            except:
                e = sys.exc_info()[0]
                print("Exception:", e)
                continue
    # print("Hits on article - {}".format(item))
    return corpus, d


def TfIdf(corpus, text):
    cv = CountVectorizer(max_df=0.85, max_features=1000, ngram_range=(1, 3))
    word_count_vector = cv.fit_transform(corpus)

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    feature_names = cv.get_feature_names()

    doc = text

    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
    return [feature_names, tf_idf_vector, doc]


# Custom Sorting Function


def compare(item1, item2):
    if len(item1[0].split()) > len(item2[0].split()):
        return -1
    elif len(item1[0].split()) < len(item2[0].split()):
        return 1
    else:
        return 0


def remove_subgrams(features):
    # Sort features based on length of the n-gram
    features = sorted(features, key=lambda x: len(x.split(" ")))

    # Store the indices of features we need to remove
    to_remove = []

    # Iterate over all features
    for i, subfeature in enumerate(features):
        for j, longerfeature in enumerate(features[i + 1:]):
            if longerfeature.find(subfeature) > -1:
                to_remove.append(i)
                # break if subfeature is a substring of longerfeature
                break

    features = pd.Series(features)
    # keep only those features that are not in to_remove
    features = features.loc[~features.index.isin(to_remove)]
    return features


def extract_topn_from_vector(feature_names, items, topn=10):
    total_words = {}
    tup = zip(items.col, items.data)
    tup = set(tup)

    for item in tup:
        total_words[feature_names[item[0]]] = item[1]

    unique_words = remove_subgrams(list(total_words.keys()))
    words = {}
    for word in unique_words:
        words[word] = total_words[word]

    words = sorted(words.items(), key=lambda x: (x[1], x[0]), reverse=True)
    words = sorted(words, key=cmp_to_key(compare))

    return dict(words[:topn])


def compute_top_words(corpus, d):
    topwords = {}
    for id_, text in corpus.items():
        tfidf = TfIdf(list(corpus.values()), text)
        keywords = extract_topn_from_vector(tfidf[0], tfidf[1].tocoo(), 20)
        d[id_]['keywords'] = list(keywords.keys())
        topwords = {**topwords, **keywords}
    return topwords


def push_into_db(d, querystring, db):
    col = querystring["geo"]
    try:
        db[col].delete_many({})
    except:
        pass
    to_push = []
    for key, value in d.items():
        id_dict = {'_id': key}
        to_push.append({**id_dict, **value})

    db[col].insert_many(to_push)
    print("Pushed into Data Base")


def is_present(all_words, word):
    words = list(all_words.keys())
    for w in words:
        if (w.find(word) > -1):
            return True
    return False


def get_top_n_words_precomputed(keywords, n, three_grams=3, two_grams=5, one_grams=12):
    top_n_words = {}
    print("keywords:", keywords)
    keywords = sorted(keywords.items(), key=lambda x: (
        x[1], x[0]), reverse=True)
    keywords = sorted(keywords, key=cmp_to_key(compare))
    top_n_words = {**top_n_words, **dict(keywords[:three_grams])}

    two_gram_idx = 0
    for i in range(len(keywords)):
        if (len(keywords[i][0].split()) == 2):
            two_gram_idx = i
            break
    while (two_grams > 0 and two_gram_idx < len(keywords)):
        word = keywords[two_gram_idx][0]
        score = keywords[two_gram_idx][1]
        if (is_present(top_n_words, word)):
            two_gram_idx += 1
        else:
            top_n_words[word] = score
            two_grams -= 1
            two_gram_idx += 1

    one_gram_idx = two_gram_idx
    for i in range(two_gram_idx, len(keywords)):
        if (len(keywords[i][0].split()) == 1):
            one_gram_idx = i
            break

    while (one_grams > 0 and one_gram_idx < len(keywords)):
        word = keywords[one_gram_idx][0]
        score = keywords[one_gram_idx][1]
        if (is_present(top_n_words, word)):
            one_gram_idx += 1
        else:
            top_n_words[word] = score
            one_grams -= 1
            one_gram_idx += 1
    # print("top_n_words:", top_n_words)
    df = pd.DataFrame(columns=['tfidf']).from_dict(
        dict(top_n_words), orient='index')
    # print("data frame:", df)
    df.columns = ['tfidf']
    df = df.sort_values(by=['tfidf'], ascending=False).head(n)

    return df


def get_link_to_article(df, d):
    links = {}
    articles = {}

    for i in range(len(df)):
        word = df.index[i]
        found = False
        for i in range(10):
            for val in d.values():
                try:
                    if word == val['keywords'][i]:
                        links[word] = val['link']
                        articles[word] = val['title']
                        found = True
                        break
                except:
                    continue
            if (found):
                break
    return articles, links


def make_dataframe_top_words(data, articles, links, summary):
    data['link'] = None
    data['title'] = None
    data['summary'] = None
    for word in data.index:
        data['link'][word] = links[word] if word in links else None
        data['title'][word] = articles[word] if word in articles else None
        data['summary'][word] = summary[word] if word in summary else None
    return data


def get_link_to_article_all(df, d):
    links = {}
    articles = {}
    summary = {}

    for i in range(len(df)):
        word = df.index[i]
        list_link = []
        list_article = []
        list_summary = []
        for i in range(20):
            for val in d.values():
                try:
                    if word == val['keywords'][i]:
                        list_link.append(val['link'])
                        list_article.append(val['title'])
                        list_summary.append(val['summary'])
                except:
                    continue
        for val in d.values():
            try:
                if val['description'].find(word) >= 0 and not (val['link'] in list_link):
                    list_link.append(val['link'])
                    list_article.append(val['title'])
                    list_summary.append(val['summary'])
            except:
                continue
        links[word] = list_link
        articles[word] = list_article
        summary[word] = list_summary
    return articles, links, summary


def push_top_word_into_db(top_words, querystring, db):
    topw = {}
    col = querystring["geo"]
    print(col)

    # Get today's date and time
    now = datetime.datetime.now()
    # id_ = now.strftime("%Y_%m_%d_%H_%M_%S")
    current_time = now.strftime("%H_%M")
    todays_date = now.strftime("%m_%d_%Y")

    entry = {}
    entry['_id'] = str(uuid.uuid4())
    entry['date'] = todays_date
    entry['data'] = top_words
    entry['time'] = current_time
    # Putting time just to test, delete later
    # entry['time'] = current_time

    db[col].insert_one(entry)
    print("Top words uploaded to database.")


def change_dataframe_to_list(data):
    words = []
    for i in range(len(data)):
        topw = {}
        topw['word'] = data.loc[i]['words']
        info = []
        for j in range(len(data.loc[i]['link'])):
            ind_json = {}
            ind_json['link'] = data.loc[i]['link'][j]
            ind_json['title'] = data.loc[i]['title'][j]
            ind_json['summary'] = data.loc[i]['summary'][j]
            info.append(ind_json)

        topw['info'] = info
        # Just in case we need tf-idf score later
        topw['tfidf'] = data.loc[i]['tfidf']
        words.append(topw)
    return words


def delete_old(querystring, db):
    col = querystring["geo"]
    past_days = 3

    docs = db[col].find({})
    list_docs = []

    for doc in docs:
        list_docs.append(doc)

    # Check if we have more than past_days elements
    if (len(list_docs) <= past_days):
        print("The document has less than 3 entries")
        return

    print(f"Number of entries right now: {len(list_docs)}")
    # list_docs = sorted(list_docs, key=lambda x: datetime.datetime.strptime(x['_id'], "%Y_%m_%d_%H_%M_%S"), reverse=True)
    list_docs = sorted(list_docs, key=lambda x: datetime.datetime.strptime(
        x['date'], "%m_%d_%Y"), reverse=True)
    list_docs = list_docs[:past_days]

    db[col].delete_many({})
    for doc in list_docs:
        db[col].insert_one(doc)

    print("Deleted extra days")


def delete_old_twitter(city):
    past_days = 3

    docs = db[city].find({})
    list_docs = []

    for doc in docs:
        list_docs.append(doc)

    # Check if we have more than past_days elements
    if (len(list_docs) <= past_days):
        print("The document has less than 3 entries")
        return

    print(f"Number of entries right now: {len(list_docs)}")
    # list_docs = sorted(list_docs, key=lambda x: datetime.datetime.strptime(x['_id'], "%Y_%m_%d_%H_%M_%S"), reverse=True)
    list_docs = sorted(list_docs, key=lambda x: datetime.datetime.strptime(
        x['date'], "%b-%d-%Y"), reverse=True)
    list_docs = list_docs[:past_days]

    db[city].delete_many({})
    for doc in list_docs:
        db[city].insert_one(doc)

    print("Deleted extra days from Tweet Data")

def vectorizer_for_kmeans(corpus, grams):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=3000, ngram_range=(grams, grams))
    X = tfidf_vectorizer.fit_transform(corpus)
    
    words = tfidf_vectorizer.get_feature_names()
    return X, words


def calculate_distance(x1, y1, a, b, c):
    dist = abs((a*x1 + b*y1 + c)) / (math.sqrt(a*a + b*b))
    return dist
    
    
def optimum_num_of_cluster(K, wcss):
    # Make the line
    a = wcss[0] - wcss[8]
    b = K[8] - K[0]
    c1 = K[0] * wcss[8]
    c2 = K[8] * wcss[0]
    c = c1 - c2
    
    # Calculate distances
    dist = []
    for k in range(9):
        dist.append(calculate_distance(K[k], wcss[k], a, b, c))
    return dist.index(max(dist))


def kmeans_top_words(corpus):
    keywords_all = {}
    
    # 1, 2, and 3 grams computed seperately
    for n_gram in range(1, 4):
        # Convert to vectors
        X, words = vectorizer_for_kmeans(list(corpus.values()), n_gram)
        
        # To map the tfidf values iwth words
        total_words = {}
        tup = zip(X.tocoo().col, X.tocoo().data)
        tup = set(tup)

        for item in tup:
            total_words[words[item[0]]] = item[1]
        
        # Do pca to get the optimal number of clusters
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.todense())
        
        # Do kmeans with range of k values
        wcss = []
        K = range(1, 10)
        for i in range(1,10):
            kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
            kmeans.fit(X_pca)
            wcss.append(kmeans.inertia_)
        
        # Get optimum number of cluster
        n_cluster = optimum_num_of_cluster(K, wcss)
        print(n_cluster)
        
        # Use this n_cluster to determine the clusters
        kmeans = KMeans(init='k-means++', n_clusters = 3, n_init = 20, n_jobs = 1, max_iter=1000, random_state=3) # n_init(number of iterations for clustering) n_jobs(number of cpu cores to use)
        kmeans.fit(X)
        # We look at the clusters generated by k-means.
        common_words = kmeans.cluster_centers_.argsort()[:,-1:-50:-1]
        
        # Get the top words from each cluster
        selected_words = []
        amount_to_pick = 2
        for num, centroid in enumerate(common_words):
            for word in centroid:
                selected_words.append(words[word])

        word_tfidf = {}
        for word in selected_words:
            try:
                word_tfidf[word] = total_words[word]
            except:
                continue
        
        # Sort by tfidf score
        top_words_ngram = sorted(word_tfidf.items(), key=lambda x: (x[1], x[0]), reverse=True)
        top_words_ngram = sorted(top_words_ngram, key=cmp_to_key(compare))
        top_words_ngram = dict(top_words_ngram[:12])
        keywords_all = {**keywords_all, **top_words_ngram}
    
    return keywords_all

'''
Endpoint for user registeration
'''


@app.route('/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        email = data['email_id']
        password = data['pass']
        first_name = data['f_name']
        last_name = data['l_name']
        zipcode = data['zip']
        id = str(uuid.uuid4())
        user_exists = Users_collections.find_one({"email_id": email})

        if not user_exists:
            user_data = {"id": id, "email_id": email, "password": generate_password_hash(password),
                         "first_name": first_name, "last_name": last_name, "zip_code": zipcode}
            print(user_data)
            # res = update_recomendations(user_data)

            # if res == False:
            #     raise 'update recommendation error'

            user_id = Users_collections.insert_one(user_data)

            resp = Response('User Registered Successfully',
                            status=201, mimetype='application/json')
        else:
            resp = Response('User already exists. Please Log in.',
                            status=202, mimetype='application/json')
        return resp
    except Exception as e:
        print("Error" + str(e))
        return Response('Backend Crash', status=500, mimetype='application/json')


'''
Endpoint for user login
'''


@app.route('/login', methods=['GET', 'POST'])
def login_user():
    try:
        data = request.get_json()
        email = data['email_id']
        password = data['pass']
        user = Users_collections.find_one({"email_id": email}, {'_id': False})
        if not user:
            resp = Response('User does not exist', 401, {
                'WWW-Authenticate': 'Basic realm ="User does not exist"'})
            return resp

        if check_password_hash(user['password'], password):
            token = jwt.encode({
                'user': user
            }, SECRET_KEY)
            print(token)
            del user['password']
            resp = Response(json.dumps({'token': token, 'user': user}), 200)
            return resp

        resp = ('Wrong Password', 403, {
            'WWW-Authenticate': 'Basic realm ="Wrong Password !!"'})
        return resp
    except Exception as e:
        print('Error' + str(e))
        return Response('Backend Crash', status=500, mimetype='application/json')


@app.route('/user', methods=['Get'])
def getUser():
    user_email = request.args.get('email_id')
    print(user_email)
    if user_email:
        user_exists = Users_collections.find_one({"email_id": user_email})
        if user_exists:
            del user_exists['password']
            del user_exists['_id']
            print(user_exists)
            return str(user_exists)

        else:
            return {
                'error': "User Not Exist",
                'code': "404"
            }


@app.route('/zipcodeChange', methods=['Put'])
def zipCodeChange():
    data = request.get_json()
    print(data)
    query = {"email_id": data['email_id']}
    user_exists = Users_collections.find_one(query)
    if user_exists:
        if FetchZipCodeDetails.getZipCodeDetails(data['zip']):
            newvalues = {"$set": {"zip_code": data['zip']}}
            Users_collections.update_one(query, newvalues)
            user_exists = Users_collections.find_one(
                {"email_id": data['email_id']})
            if user_exists:
                del user_exists['password']
                del user_exists['_id']
                print(user_exists)
                return json.dumps(user_exists)
        else:
            return {
                'error_message': "Invalid Zipcode",
                'code': 404
            }
    else:
        return {
            'error_message': "Invalid User",
            'code': 404
        }


def dumpTweetsDataToDB(tweets_data, geo):
    todays_date = datetime.datetime.today().strftime('%b-%d-%Y')
    current_time = datetime.datetime.today().strftime("%H-%M")
    tweets_data['date'] = todays_date
    tweets_data['time'] = current_time
    # print(tweets_data['date'])
    collection = geo + '_twitter'
    twitter_collections = db[collection]

    print("In here Type of tweet Data", type(tweets_data))
    try:
        twitter_collections.insert_one(tweets_data)
        return tweets_data
    except Exception as E:
        return {
            'error': "Error Here"
        }


# twitter end point
@app.route('/getTweets', methods=['Post'])
def getTweets():
    try:
        data = request.get_json()
        print(data['zip'])
        print(data['word'])
        zipCode = data['zip']
        word = data['word']
        details = FetchZipCodeDetails.getZipCodeDetails(zipCode)
        print(details[0])
        lat = details[0]['lat']
        lon = details[0]['long']
        geo = details[0]['city'] + ', ' + details[0]['state']
        todays_date = datetime.datetime.today().strftime('%b-%d-%Y')
        collection = geo + '_twitter'
        print(collection)
        twitter_collections = db[collection]
        print("Here i am")
        todays_data = twitter_collections.find({'date': todays_date})
        # print(json.dumps(todays_data))
        if todays_data.count() > 0:
            list_cur = list(todays_data)

            json_data = dumps(list_cur[0], indent=2)
            # print(json_data)
            return json_data
        else:
            tweets_data = getAllTweets(lat, lon, word)
            try:
                data_send = dumpTweetsDataToDB(tweets_data, geo)
            except Exception as e:
                # print(e)
                return {
                    'error': "Error Dumping data to DB",
                    'code': 500
                }
    except Exception as e:
        print(e)
        return {
            'error': "Error in retreiving Tweets",
            'code': 500
        }
    delete_old_twitter(collection)
    return data_send
    # return "Done"


# Script to update Tweets data
def updateTweetsData(wordlist, city):
    todays_date = datetime.datetime.today().strftime('%b-%d-%Y')
    # # details = FetchZipCodeDetails.getZipCodeDetails(zipCode)
    # print(details[0])
    # lat = details[0]['lat']
    # lon = details[0]['long']
    # geo = details[0]['city'] + ', ' + details[0]['state']
    collection = city + '_twitter'
    print(collection)
    twitter_collections = db[collection]
    details = FetchZipCodeDetails.getCityDetails(city)
    print(details)
    lat = details[0]['lat']
    lon = details[0]['long']
    print(lat)
    print(lon)
    if wordlist:
        print("Wordlist Present")
        tweets_data = getAllTweets(lat, lon, wordlist)
        try:
            twitter_collections.delete_one({'date': todays_date})
            print("Deleted Twitter Data Successfully")
            dumpTweetsDataToDB(tweets_data, city)
            print("Data successfully inserted")
        except Exception as e:
            print("Error in if ", e)
    else:
        print("Wordlist not present")
        todays_data = twitter_collections.find({'date': todays_date})
        word_list = []
        if todays_data.count() > 0:
            list_cur = list(todays_data)
            for l in list_cur:
                if l.keys() not in ['_id', 'date', 'time']:
                    word_list = l.keys()

        tweets_data = getAllTweets(lat, lon, word_list)
        try:
            twitter_collections.delete_one({'date': todays_date})
            print("Deleted Twitter Data Successfully")
            dumpTweetsDataToDB(tweets_data, city)
            print("Data successfully inserted")
        except Exception as e:
            print("Error in if ", e)


def halfHourTweetsUpdate():
    ls = db.list_collection_names()
    counter = 0
    for name in ls:
        if counter < 5:
            if 'twitter' in name:
                print(name)
                # collection = db[name]

        pass

    # print(ls)
    # print("Scheduler is alive!")


def updateNewsData(city):
    url = "https://google-news.p.rapidapi.com/v1/geo_headlines"
    headers = {
        'x-rapidapi-key': "7830cda887msh2c242f864c79e4fp1b3124jsnbcea9790abd0",
        'x-rapidapi-host': "google-news.p.rapidapi.com"
    }
    collection = db[city]
    querystring = {"lang": "en", "country": "US", "geo": city}

    # Newspaper3k
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent

    print("Getting News..")
    all_news = get_news(querystring, url, headers)
    if (len(all_news) > 2):
        print("News Fetch - Done")
    else:
        print("API Request Timeout")
        return

    # Make corpus and dictionary to upload to mongoDB
    corpus, d = make_corpus(all_news, config)
    print("Make Corpus - Done")

    # Note: this will modify d and add a new key - keywords
    # keywords_all has all the keywords of individial article with tf-idf score
    keywords_all = compute_top_words(corpus, d)
    if (keywords_all is None):
        print("Failed to compute keywords")
    else:
        print("Keywords compute - Done")

    # Push the articles along with keywords into database
    # try:
    #     push_into_db(d, querystring, db)
    # except:
    #     print("Couldn't push to database -- 1")

    # Get top n words for database along with their most relevant article link
    data = get_top_n_words_precomputed(keywords_all, 40)
    articles, links, summary = get_link_to_article_all(data, d)
    data = make_dataframe_top_words(data, articles, links, summary)
    data = data.reset_index()
    data.columns = ['words', 'tfidf', 'link', 'title', 'summary']
    print("Get top n words - Done")

    top_words = change_dataframe_to_list(data)
    wordlist = []
    for word in top_words:
        wordlist.append(word['word'])

    print(wordlist)

    todays_date = datetime.datetime.now().strftime("%m_%d_%Y")

    collection.delete_one({'date': todays_date})
    print("Deleted Successful")
    # Push top n words into database
    try:
        push_top_word_into_db(top_words, querystring, db)
        print("Updated New Data News")
        updateTweetsData(wordlist, city)
    except:
        print("Couldn't push into database -- 2")

    return


def hourlyUpdateNewsData():
    ls = db.list_collection_names()
    counter = 0

    for name in ls:
        if counter < 2:
            if 'twitter' not in name and 'user' not in name:
                updateNewsData(name)


scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(hourlyUpdateNewsData, 'interval', hours=6)
scheduler.start()

# scheduler2 = BackgroundScheduler(daemon=True)
# scheduler2.add_job(halfHourTweetsUpdate, 'interval', seconds=10)
# scheduler2.start()

atexit.register(lambda: scheduler.shutdown(wait=False))
# atexit.register(lambda: scheduler2.shutdown(wait=False))

if __name__ == '__main__':
    app.run(port=5000)
