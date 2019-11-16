from googleapiclient import discovery
from api_key import API_KEY


class CommentAnalyzer(object):
    """
    Analyzing toxicity/abusive score of an arbitrary text
    using google's perspective API (https://www.perspectiveapi.com)
    """

    def __init__(self):
        self.key = API_KEY
        self.batch_scores = []
        self.request_format = {
              'comment': {'text': ""},
              'requestedAttributes': {'TOXICITY': {}}
            }

    def _response_callback(self, response_id, response, exception):
        if exception is not None:
            print(f"The following exception happened: request id- {response_id}, exception- {exception}")
            print("Returning empty scores for the entire batch")
            self.batch_scores = []
            return self.batch_scores
        else:
            # extract model score and append it to batch scores
            model_score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            self.batch_scores.append((response_id, model_score))

    def score_batch(self, comments_batch):
        # build comment analyzer service object
        service = discovery.build(serviceName='commentanalyzer', version="v1alpha1", developerKey=self.key)

        # create batch request object
        batch_requester = service.new_batch_http_request(callback=self._response_callback)
        for comment in comments_batch:
            self.request_format["comment"]["text"] = comment

            # add request to batch using updated request body
            # only request attributeScores part of response to reduce bandwidth
            batch_requester.add(service.comments().analyze(
                body=self.request_format, fields="attributeScores"))

        batch_requester.execute()

        # once the entire batch was scored, return the results in order
        self.batch_scores.sort(key=lambda x: int(x[1]))
        print(self.batch_scores)
        self.batch_scores = []


if __name__ == '__main__':
    analyzer = CommentAnalyzer()
    comments = ["hi everyone!", "how are you boy?!", "shut up!!!!", "what do you want from me?",
             "is there anybody out there?", "is there anybody out there bitch?", "dagi", "dagadag"]
    analyzer.score_batch(comments)
