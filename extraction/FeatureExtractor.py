from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(documents, num_features=5):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names()

    dense = vectors.todense()