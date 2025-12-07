import pandas as pd

def compare_reviews_both(original_file, predicted_file):
    print("Loading files...")
    try:
        # Load the datasets
        df_true = pd.read_csv(original_file)
        df_pred = pd.read_csv(predicted_file)
    except FileNotFoundError:
        print("Error: One or both files not found. Please check filenames.")
        return

    # Check if 'review' column exists in both
    if 'review' not in df_true.columns or 'review' not in df_pred.columns:
        print("Error: The column 'review' is missing from one of the files.")
        return

    # Merge on 'id' to align the rows
    merged = pd.merge(df_true, df_pred, on='id', suffixes=('_original', '_predicted'))
    total_rows = len(merged)

    # ---------------------------------------------------------
    # 1. CALCULATE FOR FULL DATASET
    # ---------------------------------------------------------
    matches_full = merged['review_original'] == merged['review_predicted']
    correct_full = matches_full.sum()
    accuracy_full = correct_full / total_rows if total_rows > 0 else 0

    print("\n" + "=" * 40)
    print(f"1. FULL DATASET RESULTS")
    print("=" * 40)
    print(f"Total Rows:     {total_rows}")
    print(f"Matches Found:  {correct_full}")
    print(f"Accuracy:       {accuracy_full:.2%}")

    # ---------------------------------------------------------
    # 2. CALCULATE FOR FIRST 40% OF DATA
    # ---------------------------------------------------------
    limit = int(total_rows * 0.40)  # Calculate 40% count
    merged_partial = merged.head(limit)
    
    matches_partial = merged_partial['review_original'] == merged_partial['review_predicted']
    correct_partial = matches_partial.sum()
    accuracy_partial = correct_partial / limit if limit > 0 else 0

    print("\n" + "=" * 40)
    print(f"2. FIRST 40% RESULTS (First {limit} rows)")
    print("=" * 40)
    print(f"Rows Compared:  {limit}")
    print(f"Matches Found:  {correct_partial}")
    print(f"Accuracy:       {accuracy_partial:.2%}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    # Replace these with your actual file names
    compare_reviews_both('Dataset/test_with_reviews_improved.csv', 'ModelPredicts/submission_lstm_nltk.csv')