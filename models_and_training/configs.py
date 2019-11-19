import json


class BaseConfig(object):
    """
    Configuration class to store the configuration objects for various use cases.
    """

    def __init__(self, config_dict=None):
        if isinstance(config_dict, dict):
            self.from_dict(config_dict)

    def from_dict(self, parameters: dict):
        """Constructs a `Config` from a Python dictionary of parameters."""
        for key, value in parameters.items():
            self.__dict__[key] = value

        return self

    def from_json_file(self, json_file: str):
        """Constructs a `Config` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
            parameters_dict = json.loads(text)

        return self.from_dict(parameters_dict)

    def to_dict(self) -> dict:
        """Serializes this instance to a Python dictionary."""
        output = self.__dict__
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str) -> None:
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class AbuseDetectConfig(BaseConfig):
    """
    Configuration class to store the configuration of an "Abuse Detection" model.
    """
    def __init__(self, config_dict=None, bert_model_name="bert-base-uncased", label_dim=2, num_hidden_features=3):
        super().__init__(config_dict)

        if config_dict is None:
            self.bert_model_name = bert_model_name
            self.label_dim = label_dim
            self.num_hidden_features = num_hidden_features


class TrainingConfig(BaseConfig):
    """
    Configuration class to store training and data loading configurations.
    """
    def __init__(self, config_dict=None, batch_size=32, num_workers=12,
                 device="cpu", num_epochs=5, learning_rate=1e-4):
        super().__init__(config_dict)

        if config_dict is None:
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.device = device
            self.num_epochs = num_epochs
            self.learning_rate = learning_rate
