from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import find
import json
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter


def extract_features(review_words, num_features=5):
    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(review_words)

    feature_names = np.asarray(vectorizer.get_feature_names())

    # print(len(feature_names))
    #
    # print(dense[1])

    (doc, feature, val) = find(vectors)

    tfidf_values = []
    for i in range(len(doc)):
        tfidf_values.append([vectors[doc[i], feature[i]], feature_names[feature[i]]])

    sorted_values = sorted(tfidf_values, key=lambda val: val[0], reverse=True)

    all_keywords = []
    for i in range(len(sorted_values[:len(sorted_values) // 4])):
        all_keywords.append(sorted_values[i][1])

    counter = Counter()
    counter.update(all_keywords)

    print(len(counter.keys()))

    count = 0
    words_added = 0
    cut_top = 10
    cut_bot = len(counter.keys()) - (len(counter.keys()) // 10)
    wordcloud_keywords = []
    while counter.most_common()[0][1] > 0 and words_added < 150:
        common_word = counter.most_common()[0][0]
        common_count = counter.most_common()[0][1]
        # if count > cut_top:
        wordcloud_keywords.extend([common_word] * common_count)
        words_added += 1
        counter.subtract(Counter({common_word: common_count}))
        count += 1
    # return

    # Generate word cloud
    wordcloud = WordCloud(width=1920, height=1080, random_state=1, background_color='black', colormap='Pastel1',
                          collocations=False, stopwords=STOPWORDS).generate(" ".join(wordcloud_keywords))

    # Set figure size
    plt.figure(figsize=(16, 9))
    # Display image
    plt.imshow(wordcloud)
    # No axis details
    plt.axis("off")
    plt.show()


    top_keywords = []
    for i in range(len(sorted_values) // 4):
        # if sorted_values[i][1] not in top_keywords:
        top_keywords.append(sorted_values[i][1])

    print(top_keywords)