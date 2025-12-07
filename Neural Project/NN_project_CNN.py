import pandas as pd
import numpy as np
import re
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalAveragePooling1D,Dense, Dropout, Concatenate, SpatialDropout1D,BatchNormalization, Activation)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight


Models = [
    {
        "name": "CNN_Basic",
        "train_file": "new_dataset/train_optimized.csv",
        "model_file": "SavedModels/cnn_aug_basic.keras",
        "sub_file": "ModelPredicts/submission_cnn_basic.csv"
    },
    {
        "name": "CNN_NLTK",
        "train_file": "new_dataset/train_optimized_nltk.csv",
        "model_file": "SavedModels/cnn_aug_nltk.keras",
        "sub_file": "ModelPredicts/submission_cnn_nltk.csv"
    }
]

TEST_PATH = 'Dataset/test.csv'

# HYPERPARAMETERS 
MAX_WORDS = 8000        
MAX_LEN = 300           
EMBEDDING_DIM = 32      
BATCH_SIZE = 16         
EPOCHS = 40             
TEST_SIZE = 0.2         

# PREPROCESSING 
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
    if pd.isna(text):
        return ""
    
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

# MODEL 
def build_cnn_model(vocab_size, num_classes, input_length):
    inputs = Input(shape=(input_length,))
    
    embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM)(inputs)
    embedding = SpatialDropout1D(0.6)(embedding) 
    
    c1 = Conv1D(filters=16, kernel_size=3, padding='same', kernel_regularizer=l2(0.02))(embedding)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = GlobalAveragePooling1D()(c1) 
    
    c2 = Conv1D(filters=16, kernel_size=4, padding='same', kernel_regularizer=l2(0.02))(embedding)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = GlobalAveragePooling1D()(c2) 
    
    c3 = Conv1D(filters=16, kernel_size=5, padding='same', kernel_regularizer=l2(0.02))(embedding)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = GlobalAveragePooling1D()(c3)
    
    merged = Concatenate()([p1, p2, p3])
    
    dense = Dense(16, kernel_regularizer=l2(0.03))(merged) # High Regularization
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dropout = Dropout(0.6)(dense)
    
    outputs = Dense(num_classes, activation='softmax')(dropout)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

# MAIN
if __name__ == "__main__":

    os.makedirs('SavedModels', exist_ok=True)
    os.makedirs('ModelPredicts', exist_ok=True)

    print(f">>> Loading Test Data from {TEST_PATH}...")
    try:
        df_test = pd.read_csv(TEST_PATH)
        df_test['text'] = df_test['text'].fillna('').astype(str)
        df_test['clean_text'] = df_test['text'].apply(clean_text)
    except FileNotFoundError:
        print(f"Error: Test file not found at {TEST_PATH}")
        exit()

    for exp in Models:
        print("\n" + "#"*60)
        print(f"   STARTING training: {exp['name']}")
        print(f"   Dataset: {exp['train_file']}")
        print("#"*60 + "\n")

        if not os.path.exists(exp['train_file']):
            print(f"Error: File {exp['train_file']} not found. Skipping...")
            continue

        K.clear_session()

        df_train = pd.read_csv(exp['train_file'])
        df_train['text'] = df_train['text'].fillna('').astype(str)
        df_train['clean_text'] = df_train['text'].apply(clean_text)

        le = LabelEncoder()
        df_train['label_id'] = le.fit_transform(df_train['review'])
        classes = le.classes_
        
        # --- Handle Class Imbalance ---
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(df_train['label_id']),
            y=df_train['label_id']
        )
        class_weights_dict = dict(enumerate(class_weights))
        print(f">>> Class Weights Computed: {class_weights_dict}")

        print(">>> Tokenizing...")
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(df_train['clean_text'])

        X = tokenizer.texts_to_sequences(df_train['clean_text'])
        X_sub = tokenizer.texts_to_sequences(df_test['clean_text'])

        X = pad_sequences(X, maxlen=MAX_LEN, padding='post', truncating='post')
        X_sub = pad_sequences(X_sub, maxlen=MAX_LEN, padding='post', truncating='post')
        y = df_train['label_id'].values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

        print(f">>> Training {exp['name']}...")
        model = build_cnn_model(MAX_WORDS, len(classes), MAX_LEN)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
            ModelCheckpoint(exp['model_file'], monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
        ]

        model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            callbacks=callbacks,
            class_weight=class_weights_dict
        )

        print(f"\n>>> Evaluating {exp['name']}...")
        try: model.load_weights(exp['model_file'])
        except: pass
        
        y_pred = np.argmax(model.predict(X_val), axis=1)
        
        print(classification_report(y_val, y_pred, target_names=[str(c) for c in classes], zero_division=0))
        print(confusion_matrix(y_val, y_pred))

        print(f">>> Saving Submission for {exp['name']}...")
        sub_preds = np.argmax(model.predict(X_sub), axis=1)
        
        pd.DataFrame({
            'id': df_test['id'], 
            'review': le.inverse_transform(sub_preds)
        }).to_csv(exp['sub_file'], index=False)
        
        print(f"Saved: {exp['sub_file']}")

    print("\n" + "="*60)
    print("ALL models COMPLETE")
    print("="*60)