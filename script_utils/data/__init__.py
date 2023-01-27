from abc import ABC, abstractmethod

class AbstractInputLoader(ABC):

    @abstractmethod
    def train_dataset(self):
        pass

    @abstractmethod
    def test_dataset(self):
        pass

    @abstractmethod
    def audit_trigger(self):
        pass

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def model_layers(self):
        pass