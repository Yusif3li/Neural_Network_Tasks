# COMPETITION-READY PREPROCESSING & AUGMENTATION
import pandas as pd
import numpy as np
import random
import re
import nltk
from nltk.corpus import wordnet

# Ensure NLTK data (run once)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def clean_text(text):
    """Basic text cleaning: lowercase, remove links/html"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text) 
    return text

def get_synonyms(word):
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            # Only keep one-word synonyms that are different from original
            if candidate != word and len(candidate.split()) == 1:
                synonyms.add(candidate)
    return list(synonyms)

def augment_text(text, n_aug=1):
    """Augment text: Swap words and replace with synonyms"""
    words = text.split()
    if len(words) < 5: return [text] * n_aug 
    
    augmented_texts = []
    
    for _ in range(n_aug):
        new_words = words.copy()
        
        # Strategy 1: Synonym Replacement (Agresive: 40% of words)
        num_to_replace = max(1, int(len(words) * 0.4)) 
        indices = random.sample(range(len(words)), num_to_replace)
        
        for idx in indices:
            syns = get_synonyms(new_words[idx])
            if syns:
                new_words[idx] = random.choice(syns)
        
        # Strategy 2: Random Swap (Swap 2 pairs)
        if len(new_words) > 4:
            for _ in range(2):
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
        augmented_texts.append(" ".join(new_words))
        
    return augmented_texts

def process_dataset(input_file='train.csv', output_file='train_balanced.csv'):
    print(f"🔄 Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        # Fallback for folder structure
        df = pd.read_csv(f'dataset/{input_file}')
        
    initial_rows = len(df)
    
    # 1. Clean Garbage
    print("🧹 Cleaning data...")
    df.drop_duplicates(subset=['text'], inplace=True)
    df['text'] = df['text'].apply(clean_text)
    # Drop rows with < 3 words
    df = df[df['text'].apply(lambda x: len(str(x).split()) >= 3)]
    
    print(f"   Cleaned {initial_rows - len(df)} garbage rows.")
    
    # 2. Determine Target Count (Majority Class)
    class_counts = df['review'].value_counts()
    target_count = class_counts.max() 
    print(f"📊 Target for ALL classes: {target_count} samples")
    
    balanced_data = []
    
    # 3. Augmentation Loop
    for label in class_counts.index:
        class_df = df[df['review'] == label]
        current_count = len(class_df)
        balanced_data.append(class_df) # Add original data
        
        if current_count < target_count:
            needed = target_count - current_count
            print(f"   ✨ Augmenting '{label}': Need {needed} more...")
            
            source_texts = class_df['text'].tolist()
            new_texts = []
            failures = 0
            
            # FORCE LOOP: Keep going until we hit the target
            while len(new_texts) < needed:
                text = random.choice(source_texts)
                aug_text = augment_text(text)[0]
                
                # Try to add unique text
                if aug_text not in new_texts:
                    new_texts.append(aug_text)
                    failures = 0 # Reset failure counter
                else:
                    failures += 1
                
                # Safety Valve: If we fail 50 times to find a unique text, allow a duplicate
                # This prevents infinite loops on small datasets
                if failures > 50:
                    new_texts.append(aug_text) 
                    failures = 0
            
            # Add the new generated data
            aug_df = pd.DataFrame({
                'id': [99999] * len(new_texts),
                'text': new_texts,
                'review': [label] * len(new_texts)
            })
            balanced_data.append(aug_df)
    
    # 4. Save Final
    final_df = pd.concat(balanced_data).sample(frac=1, random_state=42).reset_index(drop=True)
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ SUCCESS! Saved {len(final_df):,} rows to {output_file}")
    print("\nFINAL CLASS DISTRIBUTION (Should be equal):")
    print(final_df['review'].value_counts())

if __name__ == "__main__":
    process_dataset()