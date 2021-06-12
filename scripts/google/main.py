import csv
import json
import re
import nltk
from nltk.stem import WordNetLemmatizer


def get_data(file):
    data = []

    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # Ignore non-existing reviews
            if row[2] == "nan":
                continue
            data.append(row)
    return data


def format_data(data):
    """Takes Google App Store data and formats the comments using NLTK.

    Args:
        data (List): List of reviews.

    Returns:
        List: List of reviews with comment strings being replaced by a list of keywords and phrases.
    """
    wnl = WordNetLemmatizer()

    # Dictionaries for word changes
    destroy = ["'m", "'s"]
    replace = {
        "&": "and"
    }

    for i in range(1, len(data)):
        # Replace common phrasing
        data[i][1] = data[i][1].replace("can't", "can not").replace("CAN'T", "CAN NOT").replace("Can't", "Can not")

        # Tokenize comment string
        tokens = nltk.word_tokenize(data[i][1])
        lem_tokens = []

        pos_list = nltk.pos_tag(tokens)

        # Obtain pos tag for each word token
        for k in range(len(pos_list)):
            word = pos_list[k][0]
            tag = pos_list[k][1]
            # If word has some alternate abbreviation, replace it
            # Else if word has no significant meaning, ignore it
            if word.lower() in replace:
                word = replace[word]
            elif word in destroy or not re.search("[$#a-zA-Z0-9]", word):
                continue

            # Get pos tag and see if it is valid
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None

            # If not a valid tag, add to tokens as-is. Otherwise lemmatize it.
            if not wntag:
                lem_tokens.append(word)
            else:
                lem_tokens.append(wnl.lemmatize(word, wntag))

        # Replace comment string with new word tokens listing
        data[i][1] = lem_tokens
    return data


def make_data_app_dict(data):
    """Takes Google App Store review data and creates a dictionary with apps as keys.

    Args:
        data (List): List of reviews, with each row being a review.

    Returns:
        dict: A dict with apps (string) as keys, containing all comments for each app.
    """
    output = {}

    # Ignore header row
    for row in data[1:]:
        if row[0] not in output:
            output[row[0]] = []
        output[row[0]].append(row[1:])
    return output


nltk.download("punkt")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')

print("Obtaining Data")
data = get_data("../../data/google/googleplaystore_user_reviews.csv")

print("Formatting Data")
data = format_data(data)

print("Saving Data")
with open("../../data/google/app_user_reviews.json", "w", encoding="utf-8") as outfile:
    json.dump(make_data_app_dict(data), outfile)
with open("../../data/google/user_reviews.json", "w", encoding="utf-8") as outfile:
    json.dump(data, outfile)
