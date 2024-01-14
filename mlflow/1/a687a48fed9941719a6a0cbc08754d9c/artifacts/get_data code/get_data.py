try:
    import sys

    sys.path.insert(0, "..")
    from config.tokens import youtube_api_token
except Exception:
    # My YouTube API Token
    youtube_api_token = "AIzaSyA66t4fhd9MTQfjh7z9DD0zATCRn5xMuyQ"


import requests
import json
import os
import mlflow

from pyyoutube import Api
from mlflow.tracking import MlflowClient


os.environ["MLFLOW_REGISTRY_URI"] = "/home/ml-srv/ml_pr3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

key = youtube_api_token
api = Api(api_key=key)

query = "'Machine Learning'"
video = api.search_by_keywords(q=query, search_type=["video"], count=100, limit=300)

max_result = 100
next_page_token = ""

s = 0

with mlflow.start_run():
    channels = {}
    
    for i, id_ in enumerate([x.id.videoId for x in video.items]):
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
        count = 0
        try:
            for item in data["items"]:
                count += 1
                ci = item['snippet']['topLevelComment']['snippet']
                if ci['channelId'] in channels:
                    channels[ci['channelId']] += ci['likeCount']
                else:
                    channels[ci['channelId']] = ci['likeCount']
        except Exception:
            pass
    mlflow.log_artifact(
        local_path="/home/ml-srv/ml_pr3/scripts/get_data.py",
        artifact_path="get_data code",
    )
    mlflow.end_run()


with open("/home/ml-srv/ml_pr3/datasets/data.csv", "a") as f:
    f.write("{}\n".format(s))
