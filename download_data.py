"""
download_data.py
================
ILDC dataset download karta hai HuggingFace se.
Run: python download_data.py
"""

from datasets import load_dataset
import os
import json
from tqdm import tqdm

print("=" * 50)
print("  INDIAN LEGAL AI — Dataset Download")
print("=" * 50)

# Output folder
os.makedirs("data/raw", exist_ok=True)

print("\n[1/3] ILDC dataset download ho raha hai...")
print("      (pehli baar thoda time lagega ~500MB)\n")

try:
    # ILDC Single — judgment prediction dataset
    dataset = load_dataset("Exploration-Lab/ILDC", "ILDC_single")

    print(f"\n[2/3] Dataset info:")
    print(f"      Train samples : {len(dataset['train'])}")
    print(f"      Test samples  : {len(dataset['test'])}")
    print(f"      Columns       : {dataset['train'].column_names}")

    print("\n[3/3] Saving to data/raw/ ...")

    # Save train
    train_records = []
    for item in tqdm(dataset['train'], desc="Saving train"):
        train_records.append({
            "text"  : item['text'],
            "label" : item['label'],   # 0 = dismissed, 1 = accepted
        })

    with open("data/raw/train.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    # Save test
    test_records = []
    for item in tqdm(dataset['test'], desc="Saving test"):
        test_records.append({
            "text"  : item['text'],
            "label" : item['label'],
        })

    with open("data/raw/test.json", "w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)

    print(f"\n Done! Files saved:")
    print(f"   data/raw/train.json  ({len(train_records)} cases)")
    print(f"   data/raw/test.json   ({len(test_records)} cases)")
    print(f"\nAb run karo: python preprocess.py")

except Exception as e:
    print(f"\n Error: {e}")
    print("   Internet check karo ya HuggingFace account se login karo:")
    print("   huggingface-cli login")