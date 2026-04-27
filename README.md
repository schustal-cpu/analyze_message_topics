# analyze\_message\_topics

Uses the following two Datasets for analyzing:

* formal Complaints (german):
https://fragdenstaat.de/api/v1/request/
Downloaded via get\_data\.py and stored in data/fragdenstaat\_messages\_raw.json/

* social Media Messages (english):
https://business.yelp.com/data/resources/open-dataset/
Manual Downloaded and stored in data/yelp\_messages\_raw.json

get_data.py creates a csv File with 1000 Lines per DataSet with the following format.

origin,id,created,review
"Yelp/FragdenStaat",int,time,text

analyze_data.py: Topic Identification and Sentiment Analysis based on the created CSV file

