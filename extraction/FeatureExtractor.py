from sklearn.feature_extraction.text import TfidfVectorizer
from  scipy.sparse import find
import json
import numpy as np

def extract_features(review_file="../data/google/user_reviews.json", num_features=5):
    with open(review_file) as json_reviews:
        user_reviews = json.load(json_reviews)

    review_words = []
    for review in user_reviews[1:101]:
        review_words.append(" ".join(review[1]))

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(review_words)

    feature_names = np.asarray(vectorizer.get_feature_names())

    dense = vectors.todense()

    # print(len(feature_names))
    #
    # print(dense[1])

    (doc, feature, val) = find(vectors)

    tfidf_values = []
    for i in range(len(doc)):
        tfidf_values.append([vectors[doc[i], feature[i]], feature_names[feature[i]]])

    sorted_values = sorted(tfidf_values, key=lambda val: val[0])
    print(sorted_values)

    # print(list(zip(np.matrix.tolist(vectors[doc, feature]), feature_names[feature])))

