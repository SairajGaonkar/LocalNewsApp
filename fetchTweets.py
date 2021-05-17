#!/usr/bin/env python3

# -----------------------------------------------------------------------
# twitter-search-geo
#  - performs a search for tweets close to New Cross, London,
#    and outputs them to a CSV file.
# -----------------------------------------------------------------------
import uuid

import config
from twitter import *
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')
import sys


def getSentimentScore(text):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(str(text))
    return score


def getAllTweets(lat, lon, wordlist):
    print("In fetch tweets")

    # print("word", wordlist)
    res = {}
    for word in wordlist:
        latitude = float(lat)  # geographical centre of search
        longitude = float(lon)   # geographicalntre of search
        # print("lat", latitude)
        # print("lon", longitude)
        max_range = 40             # search range in kilometres
        num_results = 5        # minimum results to obtain
        outfile = "output_tweet.csv"

        # -----------------------------------------------------------------------
        # load our API credentials
        # -----------------------------------------------------------------------
        sys.path.append(".")

        # -----------------------------------------------------------------------
        # create twitter API object
        # -----------------------------------------------------------------------
        t = Twitter(auth=OAuth(config.access_key,
                               config.access_secret,
                               config.consumer_key,
                               config.consumer_secret))

        # -----------------------------------------------------------------------
        # the twitter API only allows us to query up to 100 tweets at a time.
        # to search for more, we will break our search up into 10 "pages", each
        # of which will include 100 matching tweets.
        # -----------------------------------------------------------------------
        result_count = 0
        last_id = None
        # while result_count < num_results:
        # -----------------------------------------------------------------------
        # perform a search based on latitude and longitude
        # twitter API docs: https://dev.twitter.com/rest/reference/get/search/tweets
        # -----------------------------------------------------------------------
        query = t.search.tweets(q=word, geocode="%f,%f,%dkm" % (
            latitude, longitude, max_range), count=100, max_id=last_id, result_type="recent")
        # print(len(query))
        # print(len(query["statuses"]))
        # print(query["statuses"])
        # print(query)

        tempList = []
        text = ""
        for result in query["statuses"]:
            temp = {}
            text += result['text']
            temp_user = {}
            if len(result["entities"]["urls"]) >= 1:
                temp['url'] = result["entities"]["urls"][0]["url"]
            if len(result['user']) >=1:
                temp_user['name'] = result['user']['name']
                temp_user['profile_image_url'] = result['user']['profile_image_url']
            temp['text'] = result['text']
            temp['user'] = temp_user
            # temp['user']['profile_image_url'] = result['user']['profile_image_url']
            temp['tweet_date'] = result['created_at']
            temp['retweet_count'] = result['retweet_count']
            temp['like'] = result['favorite_count']
            tempList.append(temp)

        sentiment_score = getSentimentScore(text)
        word_object = {
            'tweets': tempList,
            'sentiment_score': sentiment_score
        }
        res['_id'] = str(uuid.uuid4())
        res[str(word)] = word_object
    # print(res)
    return res
    #
    # print("---------------------------------------------")
    # # print(query["statuses"])
    # for result in query["statuses"]:
    #     url = ''
    #     # print(result)
    #     # break
    #     # -----------------------------------------------------------------------
    #     # only process a result if it has a geolocation
    #     # -----------------------------------------------------------------------
    #     user = result["user"]["screen_name"]
    #     text = result["text"]
    #     if len(result["entities"]["urls"]) > 1:
    #         url = result["entities"]["urls"][0]["url"]
    #     # print(text)
    #     text = text.encode('ascii', 'replace')
    #     # -----------------------------------------------------------------------
    #     # now write this row to our CSV file
    #     # -----------------------------------------------------------------------
    #     row = [user, text, url]
    #     csvwriter.writerow(row)
    #     last_id = result["id"]
    #
    # # -----------------------------------------------------------------------
    # # let the user know where we're up to
    # # -----------------------------------------------------------------------
    # # print("got %d results" % result_count)
    #
    # # -----------------------------------------------------------------------
    # # we're all finished, clean up and go home.
    # # -----------------------------------------------------------------------
    # csvfile.close()
    #
    # print("written to %s" % outfile)
