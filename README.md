# Data Extraction & WordCloud Analysis
This project creates wordclouds using reviews from Google Play Store applications 
and Steam games. This is done by extracting key words from reviews and giving values 
using a TF-IDF to determine value of words.

This project is for the NanoGiants Hackathon 2021 that started on June 11th, 2021.

## Demo Instructions
1. Install the Python modules in requirements.txt using pip. This can be done using
`pip install -r requirements.txt`
2. Download the data sets from https://www.kaggle.com/lava18/google-play-store-apps and https://www.kaggle.com/tamber/steam-video-games
3. Run scripts/google/main.py
4. Run scripts/steam/main.py (This script takes a while to format the data due to the dataset size. Allow approximately 10 minutes to format.)
5. Run scripts/google/tfidfcloud.py
    - The num variable can be used to run the script on different application reviews.
6. Run scripts/steam/tfidfcloud.py
    - The num variable can be used to run the script on different game reviews.
