# IMPROVED DATA ANALYSIS CODE

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# --- CONFIGURATION: DEFINE PATHS HERE ---
DATASETS = {
    "1. BASIC Augmentation": os.path.join('dataset', 'train.csv'),
    #"2. NLTK Augmentation":  os.path.join('new_dataset', 'train_optimized_nltk.csv')
}

def count_patterns(pattern, text_series):
    """Count pattern occurrences in text series"""
    return text_series.apply(lambda x: len(re.findall(pattern, str(x)))).sum()

def analyze_dataset(name, file_path, save_plots=True):
    """Enhanced dataset analysis with visualizations"""
    print("\n" + "="*70)
    print(f"   COMPREHENSIVE ANALYSIS REPORT FOR: {name}")
    print(f"   File: {file_path}")
    print("="*70)

    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found!")
        print(f"   Current Working Directory: {os.getcwd()}")
        return None
    
    df = pd.read_csv(file_path)
    print(f"✅ Successfully loaded {len(df):,} rows, {df.shape[1]} columns")
    
    # Create analysis directory
    if save_plots:
        os.makedirs('analysis_reports', exist_ok=True)
    
    # --- 1. INTEGRITY & BASIC STATS ---
    print(f"\n📊 1. DATASET INTEGRITY")
    print(f"{'-'*40}")
    nulls = df['text'].isnull().sum()
    empties = (df['text'].str.strip() == '').sum()
    duplicates = df.duplicated(subset=['text']).sum()
    
    print(f"Missing (null) Text Rows: {nulls:,}")
    print(f"Empty Text Rows: {empties:,}")
    print(f"Duplicate Text Rows: {duplicates:,}")
    print(f"Unique Texts: {df['text'].nunique():,}")
    
    # --- 2. LENGTH ANALYSIS ---
    df['text'] = df['text'].astype(str)
    df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))
    df['char_len'] = df['text'].apply(lambda x: len(str(x)))
    
    avg_len = int(df['text_len'].mean())
    max_len = df['text_len'].max()
    p95 = int(df['text_len'].quantile(0.95))
    p99 = int(df['text_len'].quantile(0.99))
    
    print(f"\n📏 2. TEXT LENGTH ANALYSIS")
    print(f"{'-'*40}")
    print(f"Avg Words: {avg_len:,}")
    print(f"Max Words: {max_len:,}")
    print(f"95th Percentile: {p95:,} words")
    print(f"99th Percentile: {p99:,} words")
    print(f"\n📈 RECOMMENDATION: Set MAX_LEN = {p95} (covers 95% of data)")
    print(f"   Alternative: Set MAX_LEN = {max(50, min(p95, 200))} for efficiency")
    
    # Visualize length distribution
    if save_plots:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.hist(df['text_len'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(p95, color='red', linestyle='--', label=f'95th: {p95}')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title(f'{name}\nText Length Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # --- 3. NOISE & PATTERN ANALYSIS ---
    print(f"\n🔍 3. NOISE PATTERN DETECTION")
    print(f"{'-'*40}")
    
    patterns = {
        'URLs': r'http\S+|www\.\S+',
        'Non-ASCII': r'[^\x00-\x7F]',
        'Emails': r'\S+@\S+\.\S+',
        'Mentions': r'@\w+',
        'Hashtags': r'#\w+',
        'Numbers': r'\b\d+\b',
        'Special Chars': r'[^\w\s]',
        'Repeated Chars': r'(.)\1{3,}'
    }
    
    for pattern_name, pattern in patterns.items():
        count = count_patterns(pattern, df['text'])
        if count > 0:
            print(f"{pattern_name:20s}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # --- 4. VOCABULARY ANALYSIS ---
    print(f"\n📚 4. VOCABULARY ANALYSIS")
    print(f"{'-'*40}")
    
    all_text = " ".join(df['text'].str.lower())
    words = all_text.split()
    word_counter = Counter(words)
    vocab_size = len(word_counter)
    
    # Calculate coverage statistics
    sorted_words = word_counter.most_common()
    total_words = len(words)
    
    # Find how many words cover 95% of occurrences
    cumulative = 0
    words_for_95 = 0
    for i, (word, count) in enumerate(sorted_words):
        cumulative += count
        if cumulative / total_words >= 0.95:
            words_for_95 = i + 1
            break
    
    print(f"Total Words: {total_words:,}")
    print(f"Unique Words: {vocab_size:,}")
    print(f"Average word frequency: {total_words/vocab_size:.1f}")
    print(f"\n📈 VOCABULARY COVERAGE:")
    print(f"Top 100 words cover {sum([count for _, count in sorted_words[:100]])/total_words*100:.1f}%")
    print(f"Top 1000 words cover {sum([count for _, count in sorted_words[:1000]])/total_words*100:.1f}%")
    print(f"Words needed for 95% coverage: {words_for_95:,}")
    print(f"\n📈 RECOMMENDATION: Set MAX_WORDS = {min(words_for_95 * 2, vocab_size):,}")
    
    # Top words
    print(f"\n🔝 TOP 20 MOST COMMON WORDS:")
    for word, count in sorted_words[:20]:
        print(f"  {word:15s}: {count:,}")
    
    # Word cloud visualization
    if save_plots:
        plt.subplot(1, 2, 2)
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                            max_words=100).generate(all_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Words')
        plt.tight_layout()
        plt.savefig(f'analysis_reports/{name.replace(" ", "_")}_analysis.png', dpi=150)
        plt.close()
    
    # --- 5. CLASS DISTRIBUTION ---
    print(f"\n⚖️ 5. CLASS BALANCE ANALYSIS")
    print(f"{'-'*40}")
    
    class_dist = df['review'].value_counts()
    total = len(df)
    
    print(f"Total samples: {total:,}")
    print(f"Number of classes: {len(class_dist)}")
    print(f"\nClass Distribution:")
    for label, count in class_dist.items():
        percentage = count/total*100
        print(f"  {label:20s}: {count:6,} ({percentage:5.1f}%)")
    
    # Calculate imbalance metrics
    max_class = class_dist.max()
    min_class = class_dist.min()
    imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
    
    print(f"\n📊 IMBALANCE METRICS:")
    print(f"  Largest class / Smallest class: {imbalance_ratio:.1f}")
    print(f"  Standard deviation: {class_dist.std():.1f}")
    
    if imbalance_ratio > 5:
        print(f"  ⚠️  WARNING: Severe class imbalance (>5:1 ratio)")
    elif imbalance_ratio > 3:
        print(f"  ⚠️  Moderate class imbalance (3-5:1 ratio)")
    else:
        print(f"  ✅ Good class balance")
    
    # Class distribution visualization
    if save_plots:
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_dist)))
        bars = plt.bar(class_dist.index, class_dist.values, color=colors)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(f'{name}\nClass Distribution')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                    f'{int(height):,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'analysis_reports/{name.replace(" ", "_")}_classes.png', dpi=150)
        plt.close()
    
    # --- 6. TEXT QUALITY METRICS ---
    print(f"\n✨ 6. TEXT QUALITY METRICS")
    print(f"{'-'*40}")
    
    # Average sentence complexity
    avg_words_per_sentence = df['text_len'].mean()
    unique_words_ratio = vocab_size / total_words
    
    print(f"Average words per text: {avg_words_per_sentence:.1f}")
    print(f"Vocabulary richness: {unique_words_ratio*100:.2f}%")
    
    # Check for very short/long texts
    short_texts = (df['text_len'] < 3).sum()
    long_texts = (df['text_len'] > 500).sum()
    
    if short_texts > 0:
        print(f"⚠️  Texts with <3 words: {short_texts:,}")
    if long_texts > 0:
        print(f"⚠️  Texts with >500 words: {long_texts:,}")
    
    print("-" * 70)
    
    # Return summary statistics
    summary = {
        'name': name,
        'rows': len(df),
        'classes': len(class_dist),
        'vocab_size': vocab_size,
        'avg_length': avg_len,
        'p95_length': p95,
        'imbalance_ratio': imbalance_ratio,
        'recommended_max_len': p95,
        'recommended_max_words': min(words_for_95 * 2, vocab_size)
    }
    
    return summary

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"📁 Script running in: {os.getcwd()}")
    print(f"📊 Starting comprehensive dataset analysis...\n")
    
    all_summaries = []
    
    for name, path in DATASETS.items():
        summary = analyze_dataset(name, path, save_plots=True)
        if summary:
            all_summaries.append(summary)
    
    # Print comparative summary
    if all_summaries:
        print("\n" + "="*70)
        print("📋 COMPARATIVE SUMMARY")
        print("="*70)
        
        summary_df = pd.DataFrame(all_summaries)
        print(summary_df.to_string(index=False))
        
        # Save summary to CSV
        summary_df.to_csv('analysis_reports/dataset_summary.csv', index=False)
        print(f"\n✅ Analysis reports saved to 'analysis_reports/' folder")