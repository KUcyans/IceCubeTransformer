from enum import Enum

class ClassificationMode(Enum):
    MULTIFLAVOUR = (0, "Multiflavour", 3)
    TRACK_CASCADE_BINARY = (1, "TrackCascadeBinary", 2)
    SIGNAL_NOISE_BINARY = (2, "SignalNoiseBinary", 2)
    
    def __init__(self, value, name, num_classes):
        self._value_ = value
        self._name_ = name
        self._num_classes = num_classes

    @classmethod
    def from_string(cls, string):
        for mode in cls:
            if mode.name == string:
                return mode
        raise ValueError(f"Invalid classification mode: {string}")

    @classmethod
    def from_value(cls, value):
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"Invalid classification mode value: {value}")

    @property
    def value(self):
        return self._value_

    @property
    def num_classes(self):
        return self._num_classes
