try:
    import sys

    sys.path.insert(0, "..")
    from config.tokens import youtube_api_token
except Exception:
    # My YouTube API Token
    youtube_api_token = ""


import requests
import json

from pyyoutube import Api


key = youtube_api_token
api = Api(api_key=key)

query = "'Machine Learning'"
video = api.search_by_keywords(q=query, search_type=["video"], count=100, limit=300)

max_result = 100
next_page_token = ""

s = 0

for id_ in [x.id.videoId for x in video.items]:
    uri = (
        "https://www.googleapis.com/youtube/v3/commentThreads?"
        + "key={}&textFormat=plainText&"
        + "part=snippet&"
        + "videoId={}&"
        + "maxResults={}&"
        + "pageToken={}"
    )
    uri = uri.format(key, id_, max_result, next_page_token)
    content = requests.get(uri).text
    data = json.loads(content)
    try:
        for item in data["items"]:
            s += int(item["snippet"]["topLevelComment"]["snippet"]["likeCount"])
    except Exception:
        pass


with open("/home/ml-srv/ml_pr3/datasets/data.csv", "a") as f:
    f.write("{}\n".format(s))
