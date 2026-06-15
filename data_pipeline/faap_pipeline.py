import json
import os
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'full'])
    args = parser.parse_args()

    print(f"\n--- Running FAAP Decontextualization Pipeline in {args.mode.upper()} mode ---")
    
    # Load dataset
    print("Loading nfliu/decontextualization dataset...")
    split_query = 'train[:100]' if args.mode == 'test' else 'train'
    ds = load_dataset('nfliu/decontextualization', split=split_query)
    
    os.makedirs('processed_data', exist_ok=True)
    out_path = f'processed_data/faap_instructions_{args.mode}.jsonl'
    
    valid_count = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for entry in ds:
            # According to nfliu/decontextualization, the target is in annotations
            # Let's extract the first valid decontextualized sentence
            target = None
            if 'annotations' in entry and entry['annotations']:
                # The annotations field usually contains lists of human rewrites
                # If there's a 'decontextualized_sentence' field, we grab it.
                # Just in case of different schemas, we safely parse:
                ann = entry['annotations'][0] if isinstance(entry['annotations'], list) else entry['annotations']
                if isinstance(ann, dict) and 'decontextualized_sentence' in ann:
                    target = ann['decontextualized_sentence']
                elif isinstance(ann, str):
                    target = ann
            
            if not target:
                continue
                
            prompt = (
                "Rewrite the following sentence so that it can be understood strictly on its own, "
                "resolving all pronouns and adding necessary entities from the context paragraph. "
                "Output ONLY the single, autonomous fact.\n\n"
                f"Context: {entry.get('paragraph_text', '')}\n"
                f"Sentence: {entry.get('original_sentence', '')}"
            )
            
            out_obj = {
                "instruction": prompt,
                "input": "",
                "output": target
            }
            f.write(json.dumps(out_obj) + '\n')
            valid_count += 1
            
    print(f"Successfully exported {valid_count} FAAP instruction pairs to {out_path}!")

if __name__ == "__main__":
    main()
