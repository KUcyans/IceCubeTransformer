from enum import Enum

class EnergyRange(Enum):
    ER_100_GEV_10_TEV = (0, ["22010", "22013", "22016"])
    ER_10_TEV_1_PEV   = (1, ["22011", "22014", "22017"])
    ER_1_PEV_100_PEV  = (2, ["22012", "22015", "22018"])

    def __init__(self, value, subdirs):
        self._value_ = value
        self._subdirs = subdirs

    def get_subdirs(self):
        return self._subdirs