import json
from scripts.extraction.extractor import extract_features


num = 0

review_json_dict = {}
with open("../../data/google/app_user_reviews.json") as review_file:
    review_json_dict = json.load(review_file)

review_words = []
for review in review_json_dict[list(review_json_dict)[num]]:
    review_words.append(" ".join(review[0]))

extract_features(review_words)

print(list(review_json_dict)[num])