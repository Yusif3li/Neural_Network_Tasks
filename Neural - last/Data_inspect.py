# IMPROVED INSPECTION CODE

import pandas as pd
import os
import random
import numpy as np
from collections import Counter

# --- CONFIGURATION ---
DATASETS_TO_CHECK = {
    "1. BASIC Augmentation (Random Swap/Delete)": 'new_dataset/train_optimized.csv',
    "2. NLTK Augmentation (Synonym Replacement)": 'new_dataset/train_optimized_nltk.csv'
}

def get_sample_texts(df, label, n_samples=3, max_length=500):
    """Get random samples for a specific label"""
    label_texts = df[df['review'] == label]['text'].values
    if len(label_texts) == 0:
        return []
    
    samples = []
    for _ in range(min(n_samples, len(label_texts))):
        text = random.choice(label_texts)
        # Truncate if too long for display
        if len(text) > max_length:
            text = text[:max_length] + "... [truncated]"
        samples.append(text)
    return samples

def compare_augmentations(df_basic, df_nltk, label):
    """Compare augmentations between two datasets for a specific label"""
    basic_texts = df_basic[df_basic['review'] == label]['text'].tolist()[:5]
    nltk_texts = df_nltk[df_nltk['review'] == label]['text'].tolist()[:5]
    
    print(f"\n🔍 COMPARISON FOR '{label}':")
    print(f"{'='*60}")
    
    for i in range(min(3, len(basic_texts), len(nltk_texts))):
        print(f"\nSample {i+1}:")
        print(f"BASIC:  {basic_texts[i][:100]}...")
        print(f"NLTK:   {nltk_texts[i][:100]}...")
        print("-" * 40)

def inspect_dataset(name, file_path, detailed=True):
    print(f"\n{'='*70}")
    print(f"🔎 DETAILED INSPECTION REPORT FOR: {name}")
    print(f"{'='*70}")

    if not os.path.exists(file_path):
        print(f"❌ File not found at: {file_path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Please run the augmentation script first!")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded {len(df):,} rows")
        
        # Basic checks
        print(f"\n📋 BASIC CHECKS:")
        print(f"{'-'*40}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types: {df.dtypes.to_dict()}")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"\n⚠️  Missing values found:")
            for col, count in missing[missing > 0].items():
                print(f"  {col}: {count} missing")
        
        # Class distribution with percentages
        print(f"\n📊 CLASS DISTRIBUTION:")
        print(f"{'-'*40}")
        
        class_counts = df['review'].value_counts()
        total = len(df)
        
        for label, count in class_counts.items():
            percentage = (count / total) * 100
            print(f"  {label:20s}: {count:6,} ({percentage:5.1f}%)")
        
        # Calculate and display imbalance metrics
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"\n⚖️  IMBALANCE METRICS:")
        print(f"  Largest/Smallest: {imbalance_ratio:.2f}:1")
        print(f"  Standard deviation: {class_counts.std():.1f}")
        
        if imbalance_ratio > 5:
            print(f"  ⚠️  WARNING: Severe imbalance!")
        elif imbalance_ratio > 2:
            print(f"  ℹ️   Moderate imbalance")
        else:
            print(f"  ✅ Good balance")
        
        if detailed:
            # Text quality metrics
            print(f"\n✨ TEXT QUALITY METRICS:")
            print(f"{'-'*40}")
            
            # Convert to string and calculate lengths
            df['text_str'] = df['text'].astype(str)
            df['word_count'] = df['text_str'].apply(lambda x: len(x.split()))
            df['char_count'] = df['text_str'].apply(len)
            
            print(f"Average words per text: {df['word_count'].mean():.1f}")
            print(f"Average characters per text: {df['char_count'].mean():.1f}")
            print(f"Shortest text: {df['word_count'].min()} words")
            print(f"Longest text: {df['word_count'].max()} words")
            
            # Check for very short texts (potential issues)
            short_count = (df['word_count'] < 3).sum()
            if short_count > 0:
                print(f"⚠️  Texts with <3 words: {short_count:,} ({short_count/total*100:.1f}%)")
            
            # Sample texts from each class
            print(f"\n👁️  SAMPLE TEXTS FROM EACH CLASS:")
            print(f"{'-'*40}")
            
            for label in class_counts.index:
                print(f"\n📌 Class: '{label}' ({class_counts[label]:,} samples)")
                samples = get_sample_texts(df, label, n_samples=2)
                for i, sample in enumerate(samples, 1):
                    print(f"  Sample {i}: {sample[:150]}..." if len(sample) > 150 else f"  Sample {i}: {sample}")
            
            # Vocabulary analysis by class
            print(f"\n📚 VOCABULARY BY CLASS:")
            print(f"{'-'*40}")
            
            for label in class_counts.index:
                label_texts = df[df['review'] == label]['text_str']
                all_words = " ".join(label_texts).lower().split()
                unique_words = len(set(all_words))
                print(f"  {label:20s}: {unique_words:6,} unique words")
        
        # Find and show duplicates
        duplicates = df[df.duplicated(subset=['text'], keep=False)]
        if len(duplicates) > 0:
            print(f"\n⚠️  DUPLICATE TEXTS FOUND: {len(duplicates):,}")
            print(f"{'-'*40}")
            
            # Show duplicates by class
            dup_by_class = duplicates['review'].value_counts()
            for label, count in dup_by_class.head(5).items():
                print(f"  {label:20s}: {count:6,} duplicates")
            
            if len(duplicates) <= 10:  # Show all if few
                print(f"\nDuplicate examples:")
                for idx, row in duplicates.head(3).iterrows():
                    print(f"  '{row['text'][:80]}...' → {row['review']}")
        
        print(f"\n{'='*70}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def compare_all_datasets():
    """Compare all datasets side by side"""
    print(f"\n{'='*70}")
    print("🔄 COMPARING ALL DATASETS")
    print(f"{'='*70}")
    
    datasets = {}
    
    # Load all datasets
    for name, path in DATASETS_TO_CHECK.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            datasets[name] = df
            print(f"✅ Loaded {name}: {len(df):,} rows")
        else:
            print(f"❌ Missing: {name}")
    
    if len(datasets) < 2:
        print("Need at least 2 datasets to compare")
        return
    
    # Compare sizes
    print(f"\n📊 SIZE COMPARISON:")
    print(f"{'-'*40}")
    for name, df in datasets.items():
        print(f"  {name:40s}: {len(df):,} rows")
    
    # Compare class distributions
    print(f"\n📈 CLASS DISTRIBUTION COMPARISON:")
    print(f"{'-'*40}")
    
    all_labels = set()
    for df in datasets.values():
        all_labels.update(df['review'].unique())
    
    # Create comparison table
    comparison = []
    for label in sorted(all_labels):
        row = {'Class': label}
        for name, df in datasets.items():
            count = len(df[df['review'] == label])
            row[name] = count
        comparison.append(row)
    
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    
    # Show augmentation comparison for each class
    if "1. BASIC Augmentation" in datasets and "2. NLTK Augmentation" in datasets:
        print(f"\n🔄 AUGMENTATION COMPARISON:")
        print(f"{'-'*40}")
        
        for label in all_labels:
            basic_count = len(datasets["1. BASIC Augmentation"][datasets["1. BASIC Augmentation"]['review'] == label])
            nltk_count = len(datasets["2. NLTK Augmentation"][datasets["2. NLTK Augmentation"]['review'] == label])
            
            if basic_count > 0 and nltk_count > 0:
                difference = nltk_count - basic_count
                if difference != 0:
                    print(f"  {label:20s}: BASIC={basic_count:4d}, NLTK={nltk_count:4d} (diff: {difference:+d})")
        
        # Show text comparison for minority classes
        minority_classes = [label for label in all_labels 
                          if len(datasets["1. BASIC Augmentation"][datasets["1. BASIC Augmentation"]['review'] == label]) < 1000]
        
        if minority_classes:
            print(f"\n🔍 TEXT COMPARISON FOR MINORITY CLASSES:")
            for label in minority_classes[:3]:  # Compare first 3 minority classes
                compare_augmentations(datasets["1. BASIC Augmentation"], 
                                    datasets["2. NLTK Augmentation"], 
                                    label)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"🔍 Dataset Inspection Tool")
    print(f"📁 Current directory: {os.getcwd()}\n")
    
    # Option 1: Inspect individual datasets
    individual_inspection = False
    
    if individual_inspection:
        for name, path in DATASETS_TO_CHECK.items():
            df = inspect_dataset(name, path, detailed=True)
    else:
        # Option 2: Compare all datasets
        compare_all_datasets()
    
    print(f"\n✅ Inspection complete!")