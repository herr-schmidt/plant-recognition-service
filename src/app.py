from msilib.schema import Class
from flask import Flask, request
from datetime import datetime

from classifier import Classifier


app = Flask(__name__)

# global, we want to initialize it only once
classifier = Classifier()


@app.route("/")
def home():
    return "Hello, Flask!"


@app.route("/identify", methods=['POST'])
def hello_there():
    result = classifier.classify(request.get_data())
    return result
