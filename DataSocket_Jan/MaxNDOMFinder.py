import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from .EnergyRange import EnergyRange

class MaxNDOMFinder:
    def __init__(self, 
                 root_dir: str, 
                 energy_band: EnergyRange, 
                 part: int = None, 
                 shard: int = None, 
                 verbosity: int = 0):
        self.root_dir = root_dir
        self.energy_band = energy_band
        self.part = part
        self.shard = shard
        self.verbosity = verbosity
        self.subdirectories = self.energy_band.get_subdirs()

    def __call__(self) -> int:
        max_n_doms_list = [self._get_max_n_doms_for_subdirectory(subdir) for subdir in self.subdirectories]
        max_n_doms_list = [value for value in max_n_doms_list if value is not None]

        global_max_n_doms = max(max_n_doms_list, default=0)
        if self.verbosity > 0:
            print(f"Global max_n_doms across all data: {global_max_n_doms}")
        return global_max_n_doms

    def _get_max_n_doms_for_subdirectory(self, subdirectory: str) -> int:
        if self.part is not None: # only specific parts
            part_path = os.path.join(self.root_dir, subdirectory, str(self.part))
            truth_path = os.path.join(self.root_dir, subdirectory, f"truth_{self.part}.parquet")

            return self._get_max_n_doms_for_part(part_path, truth_path)
        else: # across all parts in the subdirectory
            return self._get_max_n_doms_for_entire_subdirectory(subdirectory)

    def _get_max_n_doms_for_part(self, part_path: str, truth_path: str) -> int:
        # Get the maximum `n_doms` across all shards in a part.
        if not os.path.exists(truth_path):
            if self.verbosity > 0:
                print(f"Truth file missing for {truth_path}. Skipping.")
            return None

        truth_data = pq.read_table(truth_path)

        if self.shard is not None: # only specific shards
            
            shard_filter = self._filter_shard_data(truth_data)
            return self._compute_max_n_doms(shard_filter)
        else: # across all shards in the part
            shard_files = [
                f for f in os.listdir(part_path) if f.startswith("PMTfied_") and f.endswith(".parquet")
            ]
            max_n_doms_list = []
            for shard_file in shard_files:
                shard_no = int(shard_file.split("_")[1].split(".")[0])
                self.shard = shard_no
                max_n_doms_list.append(self._get_max_n_doms_for_shard(truth_data))
            return max(max_n_doms_list, default=None)

    def _get_max_n_doms_for_shard(self, truth_data: pa.Table) -> int:
        shard_filter = self._filter_shard_data(truth_data)
        return self._compute_max_n_doms(shard_filter)

    def _get_max_n_doms_for_entire_subdirectory(self, subdirectory: str) -> int:
        # Get the maximum `n_doms` across all parts and shards in a subdirectory.
        part_dirs = [
            d for d in os.listdir(os.path.join(self.root_dir, subdirectory)) 
            if os.path.isdir(os.path.join(self.root_dir, subdirectory, d)) and d.isdigit()
        ]

        max_n_doms_list = []
        for part in part_dirs:
            self.part = int(part)
            part_path = os.path.join(self.root_dir, subdirectory, part)
            truth_path = os.path.join(self.root_dir, subdirectory, f"truth_{part}.parquet")
            max_n_doms_list.append(self._get_max_n_doms_for_part(part_path, truth_path))

        return max(max_n_doms_list, default=None)

    def _filter_shard_data(self, truth_data: pa.Table) -> pa.Table:
        shard_mask = pc.equal(truth_data.column("shard_no"), self.shard)
        return truth_data.filter(shard_mask)

    def _compute_max_n_doms(self, shard_filter: pa.Table) -> int:
        return shard_filter.column("N_doms").combine_chunks().to_numpy().max()
