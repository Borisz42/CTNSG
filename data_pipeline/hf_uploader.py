import os
import sys
from huggingface_hub import HfApi, create_repo

def upload_to_hf(token, repo_name):
    api = HfApi(token=token)
    
    print(f"Connecting to Hugging Face Hub as {repo_name}...")
    try:
        create_repo(repo_name, token=token, repo_type="dataset", exist_ok=True)
        print(f"Repository {repo_name} created or already exists.")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    readme_content = """---
license: cc-by-sa-4.0
task_categories:
- text-generation
- graph-ml
language:
- en
---

# CTNSG Graph Curriculum Dataset

This dataset contains preprocessed graphs from **WebNLG (v3.0)**, **ATOMIC**, and **Spider**. 
It is explicitly designed for the Canonical Tractable Neuro-Symbolic Generation (CTNSG) framework.

## Preprocessing
All raw data has been parsed into continuous node and edge embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
Crucially, the graphs have been mathematically canonicalized using the **Reverse Cuthill-McKee (RCM)** algorithm. 
This minimizes the adjacency-matrix bandwidth, ensuring that structurally proximal nodes have nearby indices and breaking node symmetry reproducibly.

### Topological Logic
*   **ATOMIC** events are processed using an Entity Re-use Mechanism to create multi-hop Directed Acyclic Graphs (DAGs) representing deep causal chains.
*   **Spider** text-to-SQL logic queries are structured into schema DAGs (Tables -> Columns).

## Splits
- WebNLG (v3.0): 13,211 train
- ATOMIC: 202,271 train (Causal Reasoning)
- Spider: 7,000 train (SQL Generation)

## Privacy & Legal
This dataset complies with the Right to be Forgotten via the CTNSG TRACE module architecture. 
WebNLG and Spider subsets are provided under CC BY-SA 4.0.
"""
    
    with open("README_HF.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    print("Uploading README.md Dataset Card...")
    api.upload_file(
        path_or_fileobj="README_HF.md",
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset"
    )
    
    data_file = "processed_data/ctnsg_curriculum.pt"
    if os.path.exists(data_file):
        print("Uploading serialized PyTorch tensors...")
        api.upload_file(
            path_or_fileobj=data_file,
            path_in_repo="ctnsg_curriculum.pt",
            repo_id=repo_name,
            repo_type="dataset"
        )
        print("Upload complete!")
    else:
        print(f"Error: {data_file} not found. Run download_and_preprocess.py first.")

if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        sys.exit(1)
        
    repo_name = "Borisz42/CTNSG-Graph-Curriculum"
    upload_to_hf(token, repo_name)
