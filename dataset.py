import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
import os

class CTNSGDataset(Dataset):
    """
    PyTorch Dataset that streams canonicalized CTNSG graphs directly from Hugging Face.
    """
    def __init__(self, repo_id="Borisz42/CTNSG-Graph-Curriculum", split="train"):
        # Download the .pt file from Hugging Face (will use cached version if available)
        self.file_path = hf_hub_download(repo_id=repo_id, filename="ctnsg_curriculum.pt", repo_type="dataset")
        
        print(f"Loading data from {self.file_path} into memory...")
        self.data = torch.load(self.file_path, weights_only=False) # Allow complex nested dicts
        
        # Flatten all datasets for this specific split
        self.samples = []
        for ds_name in self.data.keys():
            if split in self.data[ds_name]:
                self.samples.extend(self.data[ds_name][split])
                
        print(f"Loaded {len(self.samples)} {split} samples.")
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # Returns a dict with 'nodes' and 'edges'
        return self.samples[idx]
        
def get_ctnsg_dataloader(split="train", batch_size=32, shuffle=True):
    """
    Returns a DataLoader for the CTNSG curriculum.
    """
    dataset = CTNSGDataset(split=split)
    
    # Custom collate_fn would be needed here for graph batching
    # e.g., PyG's Batch.from_data_list if we were using torch_geometric Data objects
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Quick verification
    print("Testing CTNSGDataset initialization...")
    try:
        loader = get_ctnsg_dataloader(split="train", batch_size=2)
        batch = next(iter(loader))
        print("Success! Batch nodes shape:", batch['nodes'].shape)
    except Exception as e:
        print("Initialization failed:", e)
