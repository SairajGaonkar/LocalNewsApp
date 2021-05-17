from pymongo import MongoClient


def connectMongo():
    cluster = MongoClient(
        "mongodb://LocalNews:LNews4SoCoAtLuddy@socolab.luddy.indiana.edu/LocalNews?tls=true")
    db = cluster["LocalNews"]

    return db
