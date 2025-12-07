import pandas as pd
import os
import random

# --- CONFIGURATION ---
DATASETS_TO_CHECK = {
    "1. BASIC Augmentation (Random Swap/Delete)": 'new_dataset/train_optimized.csv',
    "2. NLTK Augmentation (Synonym Replacement)": 'new_dataset/train_optimized_nltk.csv'
}

def inspect_dataset(name, file_path):
    print(f"\n{'='*60}")
    print(f"REPORT FOR: {name}")
    print(f"{'='*60}")

    if not os.path.exists(file_path):
        print(f"❌ File not found at: {file_path}")
        print("   (Did you run that specific preprocessing script?)")
        return

    df = pd.read_csv(file_path)

    # 1. Health Check
    print(f"✅ Successfully Loaded.")
    print(f"Total Rows: {len(df)}")
    
    # 2. Balance Check
    print("\n--- Class Distribution ---")
    print(df['review'].value_counts())

    # 3. Quality Check (The Eye Test)
    print("\n--- SNEAK PEEK: 'Very bad' Reviews ---")
    # We look at 'Very bad' because it required the most augmentation
    augmented_reviews = df[df['review'] == 'Very bad']['text'].values
    
    # Show 3 examples
    for i in range(3):
        print(f"\nExample {i+1}:")
        text = random.choice(augmented_reviews)
        print(text)
        print("-" * 30)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    for name, path in DATASETS_TO_CHECK.items():
        inspect_dataset(name, path)