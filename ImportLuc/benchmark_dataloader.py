import time
from src.dataset import PMTfiedDatasetPyArrow
from torch.utils.data import DataLoader
from src.dataloader import custom_collate_fn

train_path_truth = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22011/truth_1.parquet"]

dataset = PMTfiedDatasetPyArrow(
    truth_paths=train_path_truth,
    selection=None,
    )

batch_size = 64
num_workers = 8

dataloader_new = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn, persistent_workers=True)

def main():

    # Benchmark time to load data
    print('Start iterating over the dataloader batches')
    print('Batch size:', batch_size)
    print('Number of workers:', num_workers)
    print("")
    start = time.time()

    for i, batch in enumerate(dataloader_new):
        time_per_batch = (time.time() - start) / (i + 1)
        print(f"{i} - Batches per second: {1 / time_per_batch:.2f}")
        if i == 1000:
            end = time.time()
            print(f"Time to load 1000 batches from new dataloader: {end - start:.2f}s")
            time_per_batch = (end - start) / 1000
            print(f"Time per batch: {time_per_batch:.2f}s")
            print(f"Batches per second: {1 / time_per_batch:.2f}")
            break

if __name__ == "__main__":
    main()
