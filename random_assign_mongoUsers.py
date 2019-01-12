from pymongo import MongoClient
from random import randint
from bson.json_util import dumps
# import json

client = MongoClient(port=27017)
db = client.weare

for doc in db.students.find({}):
    # obj['group'] =
    # j_doc = dumps(doc)
    db.students.update_one({'Email': doc['Email']}, {'$set': {
        'group': randint(1, int(381/15))
    }})


for doc in db.users.find({}):
    # obj['group'] =
    # j_doc = dumps(doc)
    db.users.update_one({'email': doc['email']}, {'$set': {
        'group': randint(1, int(43/10))
    }})

