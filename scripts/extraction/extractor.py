from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import find
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

"""
:param review_words: Takes in a list of lists of strings. These strings should be individual reviews.
"""
def extract_features(review_words):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(review_words)
    feature_names = np.asarray(vectorizer.get_feature_names())

    (doc, feature, val) = find(vectors)

    tfidf_values = []
    for i in range(len(doc)):
        tfidf_values.append([vectors[doc[i], feature[i]], feature_names[feature[i]]])
    sorted_values = sorted(tfidf_values, key=lambda value: value[0], reverse=True)

    all_keywords = {}
    for i in range(len(sorted_values)):
        if sorted_values[i][1] in all_keywords.keys():
            all_keywords[sorted_values[i][1]].append(sorted_values[i][0])
        else:
            all_keywords[sorted_values[i][1]] = [sorted_values[i][0]]

    generate_wordcloud(all_keywords)


def generate_wordcloud(wordcloud_keywords):
    keywords = []
    for key in wordcloud_keywords.keys():
        keywords.extend([key] * int((wordcloud_keywords[key][0] * 100)))

    # Generate word cloud
    wordcloud = WordCloud(width=1920, height=1080, random_state=1, background_color='black', colormap='Pastel1',
                          collocations=False, stopwords=STOPWORDS).generate(" ".join(keywords))

    # Set figure size
    plt.figure(figsize=(16, 9))
    # Display image
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
