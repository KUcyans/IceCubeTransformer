from enum import Enum

class EnergyRange(Enum):
    ER_100_GEV_10_TEV = {
        "nu_e": "22013",
        "nu_mu": "22010",
        "nu_tau": "22016"
    }
    ER_10_TEV_1_PEV = {
        "nu_e": "22014",
        "nu_mu": "22011",
        "nu_tau": "22017"
    }
    ER_1_PEV_100_PEV = {
        "nu_e": "22015",
        "nu_mu": "22012",
        "nu_tau": "22018"
    }

    def get_subdirs_energy(self) -> list:
        """Return all subdirectories of the given energy range."""
        return list(self.value.values())

    @classmethod
    def get_subdirs_flavour(cls, flavour: str) -> list:
        """Return all subdirectories of the given flavour across all energy ranges."""
        flavour = flavour.lower()
        return [energy_range.value[flavour] for energy_range in cls]

    @staticmethod
    def get_subdir(energy_range: 'EnergyRange', flavour: str) -> str:
        """Return the subdirectory for the given energy range and neutrino flavour."""
        flavour = flavour.lower()
        return energy_range.value[flavour]
    
    

