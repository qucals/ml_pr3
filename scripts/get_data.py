import sys

sys.path.insert(0, "..")
import config.tokens as tokens

import requests
import json

from pyyoutube import Api


key = tokens.youtube_api_token
api = Api(api_key=key)

query = "'Mission Impossible'"
video = api.search_by_keywords(q=query, search_type=["video"], count=10, limit=30)

max_result = 100
next_page_token = ""

s = 0

for id_ in [x.id.videoId for x in video.items]:
    uri = "https://www.googleapis.com/youtube/v3/commentThreads?" + \
            "key={}&textFormat=plainText&" + \
            "part=snippet&" + \
            "videoId={}&" + \
            "maxResults={}&" + \
            "pageToken={}"
    uri = uri.format(key, id_, max_result, next_page_token)
    content = requests.get(uri).text
    data = json.loads(content)
    for item in data['items']:
        s += int(item['snippet']['topLevelComment']['snippet']['likeCount'])
        
        
with open('/home/ml-srv/ml_pr3/datasets/data.csv', 'a') as f:
    f.write("{}\n".format(s))