import json


class AbuseDetectConfig(object):
    """
    Configuration class to store the configuration of an "Abuse Detection" model.
    """
    def __init__(self, config_dict_or_json=None, bert_model_name="bert-base-uncased", label_dim=2, num_hidden_features=3):
        if config_dict_or_json is None:
            self.bert_model_name = bert_model_name
            self.label_dim = label_dim
            self.num_hidden_features = num_hidden_features

        elif isinstance(config_dict_or_json, dict):
            for key, value in config_dict_or_json.items():
                self.__dict__[key] = value
        else:
            raise ValueError("First argument must be None (data is given explicitly in c'tor"
                             "or an explict dictionary object")

    @classmethod
    def from_dict(cls, parameters: dict):
        """Constructs a `Config` from a Python dictionary of parameters."""
        return AbuseDetectConfig(parameters)

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
            parameters_dict = json.loads(text)

        return cls.from_dict(parameters_dict)
