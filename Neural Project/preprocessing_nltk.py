import pandas as pd
import numpy as np
import re
import random
import os
import nltk

# --- 1. FORCE INSTALL NLTK DATA ---
print(">>> Downloading NLTK Dictionaries (One-time setup)...")
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    from nltk.corpus import wordnet
    print(">>> NLTK Ready! Synonym Replacement is ACTIVE.")
    HAS_NLTK = True
except Exception as e:
    print(f"ERROR: Could not download NLTK data: {e}")
    print("Please run in terminal: pip install nltk")
    exit()

# --- 2. CONFIGURATION ---
INPUT_FILE = 'Dataset/train.csv'
OUTPUT_DIR = 'new_dataset'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'train_optimized_nltk.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target: Balance all classes to the size of the largest class
TARGET_COUNT = 2469  

# --- 3. CLEANING LOGIC ---
EMOJI_DICT = {
    ":)": " happy ", ":-)": " happy ", ":D": " happy ", "xD": " happy ", 
    "😊": " happy ", "😍": " loved ", "👍": " good ", "❤": " love ",
    ":(": " sad ", ":-(": " sad ", ":'(": " crying ", "😭": " crying ",
    "😡": " angry ", "🤬": " angry ", "👎": " bad ", "😞": " sad ",
    "😩": " sad ", "😫": " sad "
}

CONTRACTIONS = {
    "don't": "do not", "can't": "cannot", "won't": "will not", 
    "i'm": "i am", "it's": "it is", "he's": "he is", "she's": "she is",
    "that's": "that is", "what's": "what is", "where's": "where is",
    "there's": "there is", "who's": "who is", "how's": "how is",
    "let's": "let us", "didn't": "did not", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "you're": "you are", "we're": "we are", "they're": "they are",
    "you've": "you have", "we've": "we have", "they've": "they have",
    "you'll": "you will", "we'll": "we will", "they'll": "they will",
    "i've": "i have", "i'll": "i will"
}

def normalize_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def convert_emojis(text):
    for emoji, word in EMOJI_DICT.items():
        text = text.replace(emoji, word)
    return text

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text)
    text = convert_emojis(text)
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r"\b" + contraction + r"\b", expansion, text)
    text = normalize_repeated_chars(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# --- 4. NLTK AUGMENTATION ENGINE ---
class DataAugmenter:
    def __init__(self):
        pass

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                # Keep only single words that are letters only
                if synonym != word and synonym.isalpha():
                    synonyms.add(synonym)
        return list(synonyms)

    def synonym_replacement(self, words, n):
        if n <= 0: return words
        new_words = words.copy()
        random_word_list = list(set(words))
        random.shuffle(random_word_list)
        num_replaced = 0
        
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n: break
            
        return new_words

    def generate_augmentations(self, text, n_aug=1):
        clean = clean_text(text)
        words = clean.split()
        if len(words) < 5: return [clean]
        
        augmented_texts = [clean]
        # Replace up to 15% of words
        n_change = max(1, int(0.15 * len(words)))
        
        for _ in range(n_aug):
            # Create a variation using Synonyms
            new_sent = self.synonym_replacement(words, n_change)
            augmented_texts.append(" ".join(new_sent))
            
        return list(set(augmented_texts))

# --- 5. EXECUTION ---
if __name__ == "__main__":
    print(f">>> Processing with NLTK Augmentation...")
    df = pd.read_csv(INPUT_FILE)
    aug = DataAugmenter()
    
    processed_rows = []
    groups = df.groupby('review')
    
    for label, group in groups:
        count = len(group)
        print(f" - Class '{label}': {count} rows found.")
        
        # Add originals
        for txt in group['text']:
            processed_rows.append({'text': clean_text(txt), 'review': label})
        
        # Augment Minority Classes
        if count < TARGET_COUNT:
            needed = TARGET_COUNT - count
            print(f"   -> Generating {needed} NLTK variations...")
            
            texts = group['text'].tolist()
            generated = 0
            while generated < needed:
                txt = random.choice(texts)
                # Generate 2 variations per text
                new_variations = aug.generate_augmentations(txt, n_aug=2)
                for new_txt in new_variations[1:]: 
                    processed_rows.append({'text': new_txt, 'review': label})
                    generated += 1
                    if generated >= needed: break
    
    df_optimized = pd.DataFrame(processed_rows)
    df_optimized = df_optimized.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n>>> New Class Distribution:")
    print(df_optimized['review'].value_counts())
    
    df_optimized.to_csv(OUTPUT_FILE, index=False)
    print(f"\n>>> Saved NLTK-Optimized data to: {OUTPUT_FILE}")