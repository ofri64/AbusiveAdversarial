import json
import requests


class CommentAnalyzer(object):

    def __init__(self):
        self.key = ""
        self.url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={self.key}"

    def score(self, text):
        request_data = {
            'comment': {'text': text},
            'languages': ['en'],
            'requestedAttributes': {'TOXICITY': {}}
        }
        response = requests.post(url=self.url, data=json.dumps(request_data))
        response_dict = json.loads(response.content)
        score = response_dict["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

        return score
