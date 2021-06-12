import csv
import json
import nltk
from nltk.stem import WordNetLemmatizer

import extraction.FeatureExtractor

def get_data(file):
    data = []

    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[2] == "nan":
                continue
            data.append(row)
    return data


def format_data(data):
    wnl = WordNetLemmatizer()
    for i in range(1, len(data)):
        tokens = nltk.word_tokenize(data[i][1])
        k = 0
        while k < len(tokens):
            if not tokens[k].isalnum():
                del tokens[k]
            else:
                k += 1

        data[i][1] = tokens
    return data


nltk.download("punkt")
nltk.download("wordnet")

print("Obtaining Data")
data = get_data("../../data/google/googleplaystore_user_reviews.csv")

print("Formatting Data")
data = format_data(data)

with open("../../data/google/user_reviews.json", "w", encoding="utf-8") as outfile:
    json.dump(data, outfile)

extraction.FeatureExtractor.extract_features("../../data/google/user_reviews.json")