"""
This script creates an emoji embedding of Danish, Norwegian, Swedish and
English Tweets.


The script is split into 

"""
import json

import pandas as pd

from transformers import AutoTokenizer, TFAutoModel


tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = TFAutoModel.from_pretrained("bert-base-multilingual-cased")
res = tokenizer("Hello world!")


tokenizer.tokenize("Dette er en dansk sætning URL")

for i in [101, 31178, 11356, 106, 102]:
    [101, 31178, 11356, 106, 102]
outputs = model(inputs)
len(outputs)

from urllib.parse import urlparse
urlparse('not a website')
urlparse('wwww.imc.com')

import re

myString = "This is my tweet check it out www.example.com/blah"

print(re.search("(?P<url>https?://[^\s]+)", myString).group("url"))