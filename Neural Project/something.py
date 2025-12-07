import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

def generate_reviews_improved():
    # 1. Load the datasets
    print("Loading datasets...")
    try:
        train_df = pd.read_csv('Dataset/train.csv')
        test_df = pd.read_csv('Dataset/test.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Vectorize the text data (Feature Engineering)
    print("Processing text data...")
    # ngram_range=(1, 2) means we look at single words AND pairs of words
    # e.g., "very good" is treated as a feature, not just "very" and "good"
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
    
    # Fit on train, transform both
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    y_train = train_df['review']

    # 3. Train a Real Machine Learning Model
    print("Training Logistic Regression model...")
    # Logistic Regression is generally much better for text classification than simple centroids
    model = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
    model.fit(X_train, y_train)

    # 4. Predict Reviews for Test Data
    print("Predicting reviews...")
    predicted_labels = model.predict(X_test)

    # 5. Create Final DataFrame
    test_df['review'] = predicted_labels
    
    # Add word count column
    test_df['word_count'] = test_df['text'].apply(lambda x: len(str(x).split()))

    # Select columns to match training format
    final_output = test_df[['id', 'text', 'review', 'word_count']]

    # 6. Save to CSV
    output_filename = 'test_with_reviews_improved.csv'
    final_output.to_csv(output_filename, index=False)
    print(f"Success! File saved as: {output_filename}")
    print(final_output.head())

if __name__ == "__main__":
    generate_reviews_improved()