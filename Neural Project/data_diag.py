import pandas as pd
import numpy as np
import re
import os
from collections import Counter

# --- CONFIGURATION: DEFINE PATHS HERE ---
# Based on your run logs, these should be correct.
# If your photo shows something different, update these lines!
DATASETS = {
    "1. BASIC Augmentation": os.path.join('new_dataset', 'train_optimized.csv'),
    "2. NLTK Augmentation":  os.path.join('new_dataset', 'train_optimized_nltk.csv')
}

def count_patterns(pattern, text_series):
    return text_series.apply(lambda x: len(re.findall(pattern, str(x)))).sum()

def analyze_dataset(name, file_path):
    print("\n" + "="*60)
    print(f"   ANALYSIS REPORT FOR: {name}")
    print(f"   Looking at: {file_path}")
    print("="*60)

    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found!")
        print(f"   Current Working Directory: {os.getcwd()}")
        print(f"   Please check if the folder '{os.path.dirname(file_path)}' exists.")
        return

    df = pd.read_csv(file_path)
    print(f"✅ Successfully loaded {len(df)} rows.")

    # --- 1. INTEGRITY ---
    nulls = df['text'].isnull().sum()
    print(f"\n--- 1. INTEGRITY CHECK ---")
    print(f"Missing Text Rows: {nulls}")

    # --- 2. LENGTH ANALYSIS ---
    # Ensure text is string (crucial for augmented data)
    df['text'] = df['text'].astype(str)
    df['text_len'] = df['text'].apply(lambda x: len(x.split()))
    
    avg_len = int(df['text_len'].mean())
    max_len = df['text_len'].max()
    p95 = int(df['text_len'].quantile(0.95))
    p99 = int(df['text_len'].quantile(0.99))
    
    print(f"\n--- 2. LENGTH ANALYSIS (Words) ---")
    print(f"Avg Length: {avg_len}")
    print(f"Max Length: {max_len}")
    print(f"95% of reviews are under: {p95} words")
    print(f" -> RECOMMENDATION: Set MAX_LEN to approx {p95}")

    # --- 3. NOISE HUNTING ---
    url_count = count_patterns(r'http\S+|www\.\S+', df['text'])
    non_ascii_count = count_patterns(r'[^\x00-\x7F]', df['text'])

    print(f"\n--- 3. NOISE PATTERNS ---")
    print(f"Rows with URLs:       {url_count}")
    print(f"Rows with Non-ASCII:  {non_ascii_count}")

    # --- 4. VOCABULARY ANALYSIS ---
    all_text = " ".join(df['text'].str.lower())
    words = all_text.split()
    unique_words = Counter(words)
    vocab_size = len(unique_words)
    
    # Recommendation: Cover ~90% of words for NLTK since synonyms add diversity
    rec_vocab = int(vocab_size * 0.90) 
    
    print(f"\n--- 4. VOCABULARY CHECK ---")
    print(f"Total Unique Words: {vocab_size}")
    print(f" -> RECOMMENDATION: Set MAX_WORDS to approx {rec_vocab} (or {vocab_size} if you have memory)")

    # --- 5. CLASS IMBALANCE ---
    print(f"\n--- 5. CLASS DISTRIBUTION ---")
    print(df['review'].value_counts())
    
    print("-" * 60)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"Script running in: {os.getcwd()}")
    for name, path in DATASETS.items():
        analyze_dataset(name, path)