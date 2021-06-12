import json
import scripts.extraction as extractor

num = 1

review_json_dict = {}
with open("../../data/steam/game_steam_reviews.json") as review_file:
    review_json_dict = json.load(review_file)

review_words = []
for review in review_json_dict[list(review_json_dict)[num]]:
    review_words.append(" ".join(review[-1]))

extractor.extract_features(review_words)

print(list(review_json_dict)[num])