import pandas as pd
import numpy as np
import re
import random
import os

# Try importing NLTK
try:
    import nltk
    from nltk.corpus import wordnet
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    HAS_NLTK = True
except:
    HAS_NLTK = False
    print("WARNING: NLTK not found. Augmentation limited.")

# PATHS
INPUT_FILE = 'Dataset/train.csv'
OUTPUT_DIR = 'new_dataset'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'train_optimized.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CONFIG
TARGET_COUNT = 2469  
MAX_WORDS_TO_KEEP = 43000 

# --- 1. ADVANCED DICTIONARIES ---

# Manual Emoji Dictionary (The "Secret Sauce" for Sentiment)
EMOJI_DICT = {
    # Happy
    ":)": " happy ", ":-)": " happy ", ":D": " happy ", "xD": " happy ", 
    "😊": " happy ", "😍": " loved ", "👍": " good ", "❤": " love ",
    # Sad/Angry
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

# --- 2. ADVANCED CLEANING FUNCTIONS ---

def normalize_repeated_chars(text):
    """
    Converts 'goooood' -> 'good', 'baaaad' -> 'bad'
    Replaces any character repeated more than 2 times with 2 instances.
    """
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def convert_emojis(text):
    """
    Replaces emojis with their text equivalent so the model can read them.
    """
    for emoji, word in EMOJI_DICT.items():
        text = text.replace(emoji, word)
    return text

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text)
    
    # 1. Handle Emojis FIRST (Before removing non-ascii)
    text = convert_emojis(text)
    
    text = text.lower()
    
    # 2. Expand Contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r"\b" + contraction + r"\b", expansion, text)
        
    # 3. Fix Repeated Characters (The "loooove" fix)
    text = normalize_repeated_chars(text)
    
    # 4. Standard Cleanup
    text = re.sub(r'http\S+|www\.\S+', '', text) # URLs
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # Non-ASCII
    text = re.sub(r'[^a-z\s]', '', text)         # Letters only
    
    return re.sub(r'\s+', ' ', text).strip()

# --- 3. AUGMENTATION CLASS ---
class DataAugmenter:
    def __init__(self):
        self.p_del = 0.1 

    def get_synonyms(self, word):
        if not HAS_NLTK: return []
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in 'abcdefghijklmnopqrstuvwxyz '])
                if synonym != word and synonym:
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

    def random_swap(self, words, n):
        if n <= 0 or len(words) < 2: return words
        new_words = words.copy()
        for _ in range(n):
            idx1 = random.randint(0, len(new_words)-1)
            idx2 = idx1
            while idx1 == idx2: idx2 = random.randint(0, len(new_words)-1)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return new_words

    def generate_augmentations(self, text, n_aug=1):
        clean = clean_text(text)
        words = clean.split()
        if len(words) < 5: return [clean] 
        
        augmented_texts = [clean]
        n_change = max(1, int(0.1 * len(words)))
        
        for _ in range(n_aug):
            augmented_texts.append(" ".join(self.synonym_replacement(words, n_change)))
            augmented_texts.append(" ".join(self.random_swap(words, n_change)))
            
        return list(set(augmented_texts)) 

# --- 4. EXECUTION ---
if __name__ == "__main__":
    print(f">>> Processing Data with ADVANCED Cleaning...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    aug = DataAugmenter()
    
    processed_rows = []
    groups = df.groupby('review')
    
    for label, group in groups:
        count = len(group)
        print(f" - Class '{label}': {count} rows found.")
        
        # Add originals with Advanced Cleaning
        for txt in group['text']:
            processed_rows.append({'text': clean_text(txt), 'review': label})
        
        # Augment
        if count < TARGET_COUNT:
            needed = TARGET_COUNT - count
            print(f"   -> Augmenting {needed} samples...")
            
            texts = group['text'].tolist()
            generated = 0
            while generated < needed:
                txt = random.choice(texts)
                new_variations = aug.generate_augmentations(txt, n_aug=1)
                for new_txt in new_variations[1:]: 
                    processed_rows.append({'text': new_txt, 'review': label})
                    generated += 1
                    if generated >= needed: break
                    
    df_optimized = pd.DataFrame(processed_rows)
    df_optimized = df_optimized.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n>>> New Class Distribution:")
    print(df_optimized['review'].value_counts())
    
    df_optimized.to_csv(OUTPUT_FILE, index=False)
    print(f"\n>>> Saved Optimized Data to: {OUTPUT_FILE}")