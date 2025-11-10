"""
Legal Clause Semantic Similarity - Data Preprocessing
This module handles loading, pairing, preprocessing, and tokenization of legal clause data.
"""

import os
import pandas as pd
import numpy as np
import json
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set random seed for reproducibility
np.random.seed(42)


def train_test_split(*arrays, test_size=0.15, random_state=None, stratify=None):
    """
    Custom train_test_split implementation using only numpy/pandas.
    Replaces sklearn.model_selection.train_test_split to avoid sklearn dependency.
    
    Args:
        *arrays: Arrays to split (same length)
        test_size: Proportion of test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        stratify: Array for stratified splitting (maintains class distribution)
        
    Returns:
        Split arrays: (train1, test1, train2, test2, ...) for each input array
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
    
    # Set random seed if provided (save current state to restore later if needed)
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(arrays[0])
    
    # Validate all arrays have same length
    for i, array in enumerate(arrays):
        if len(array) != n_samples:
            raise ValueError(f"All arrays must have same length. Array {i} has length {len(array)}")
    
    # Calculate test size
    n_test = max(1, int(n_samples * test_size))  # Ensure at least 1 sample in test set
    if n_test >= n_samples:
        n_test = max(1, n_samples - 1)
    n_train = n_samples - n_test
    
    if stratify is not None:
        # Stratified split: maintain class distribution
        stratify = np.asarray(stratify)
        if len(stratify) != n_samples:
            raise ValueError("stratify array must have same length as input arrays")
        
        unique_classes = np.unique(stratify)
        
        # Calculate test size for each class
        test_indices = []
        train_indices = []
        
        for cls in unique_classes:
            # Get indices for this class
            cls_indices = np.where(stratify == cls)[0]
            n_cls = len(cls_indices)
            
            if n_cls < 2:
                # If only one sample, put it in train
                train_indices.extend(cls_indices)
                continue
            
            # Calculate how many samples for test (at least 1 if possible)
            n_cls_test = max(1, int(n_cls * test_size)) if n_cls > 1 else 0
            if n_cls_test >= n_cls:
                n_cls_test = max(1, n_cls - 1) if n_cls > 1 else 0
            
            # Shuffle indices for this class
            cls_indices_shuffled = cls_indices.copy()
            np.random.shuffle(cls_indices_shuffled)
            
            # Split for this class
            if n_cls_test > 0:
                cls_test_indices = cls_indices_shuffled[:n_cls_test]
                cls_train_indices = cls_indices_shuffled[n_cls_test:]
            else:
                cls_test_indices = []
                cls_train_indices = cls_indices_shuffled
            
            test_indices.extend(cls_test_indices)
            train_indices.extend(cls_train_indices)
        
        # Convert to numpy arrays and shuffle
        test_indices = np.array(test_indices, dtype=np.int64)
        train_indices = np.array(train_indices, dtype=np.int64)
        np.random.shuffle(test_indices)
        np.random.shuffle(train_indices)
    else:
        # Random split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    # Split all arrays using the same indices
    result = []
    for array in arrays:
        array = np.asarray(array)
        result.append(array[train_indices])
        result.append(array[test_indices])
    
    return tuple(result)

def load_data(archive_dir='archive'):
    """
    Load all CSV files from archive directory and combine into a single DataFrame.
    
    Args:
        archive_dir (str): Path to directory containing CSV files
        
    Returns:
        pd.DataFrame: Combined DataFrame with columns 'clause_text' and 'clause_type'
    """
    print(f"Loading CSV files from {archive_dir}...")
    all_data = []
    
    # Get all CSV files in archive directory
    csv_files = [f for f in os.listdir(archive_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        file_path = os.path.join(archive_dir, csv_file)
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Standardize column names
            # Check if columns exist and rename if needed
            if 'clause_text' not in df.columns:
                # Try to find text column (could be first column or named differently)
                text_col = df.columns[0] if len(df.columns) > 0 else None
                if text_col:
                    df = df.rename(columns={text_col: 'clause_text'})
            
            if 'clause_type' not in df.columns:
                # Extract clause type from filename (remove .csv extension)
                clause_type = os.path.splitext(csv_file)[0]
                df['clause_type'] = clause_type
            elif df['clause_type'].isna().any():
                # Fill missing clause_type with filename
                clause_type = os.path.splitext(csv_file)[0]
                df['clause_type'] = df['clause_type'].fillna(clause_type)
            
            # Ensure we have both required columns
            if 'clause_text' in df.columns and 'clause_type' in df.columns:
                # Remove rows with missing text
                df = df.dropna(subset=['clause_text'])
                df = df[df['clause_text'].str.strip() != '']
                all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not process {csv_file}: {e}")
            continue
    
    # Combine all DataFrames
    if not all_data:
        raise ValueError("No valid data found in archive directory")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} total clauses from {len(all_data)} files")
    print(f"Unique clause types: {combined_df['clause_type'].nunique()}")
    print(f"Clause types distribution:\n{combined_df['clause_type'].value_counts().head(10)}")
    
    return combined_df


def create_pairs(df, num_positive_pairs=None, num_negative_pairs=None):
    """
    Generate positive and negative pairs from the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with 'clause_text' and 'clause_type' columns
        num_positive_pairs (int): Number of positive pairs to generate (None = use all possible)
        num_negative_pairs (int): Number of negative pairs to generate (None = match positive)
        
    Returns:
        pd.DataFrame: DataFrame with columns 'text1', 'text2', 'label'
    """
    print("\nCreating clause pairs...")
    pairs = []
    
    # Group by clause_type
    grouped = df.groupby('clause_type')
    
    # Generate positive pairs (same clause_type)
    positive_pairs = []
    for clause_type, group in grouped:
        texts = group['clause_text'].tolist()
        # Create pairs within the same type
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                positive_pairs.append({
                    'text1': texts[i],
                    'text2': texts[j],
                    'label': 1
                })
    
    print(f"Generated {len(positive_pairs)} positive pairs")
    
    # Sample positive pairs if specified
    if num_positive_pairs and len(positive_pairs) > num_positive_pairs:
        indices = np.random.choice(len(positive_pairs), num_positive_pairs, replace=False)
        positive_pairs = [positive_pairs[i] for i in indices]
    
    # Generate negative pairs (different clause_types)
    if num_negative_pairs is None:
        num_negative_pairs = len(positive_pairs)
    
    negative_pairs = []
    clause_types = df['clause_type'].unique().tolist()
    texts_by_type = {ct: df[df['clause_type'] == ct]['clause_text'].tolist() 
                     for ct in clause_types}
    
    attempts = 0
    max_attempts = num_negative_pairs * 10
    
    while len(negative_pairs) < num_negative_pairs and attempts < max_attempts:
        # Randomly select two different clause types
        type1, type2 = np.random.choice(clause_types, 2, replace=False)
        
        # Randomly select one text from each type
        if len(texts_by_type[type1]) > 0 and len(texts_by_type[type2]) > 0:
            text1 = np.random.choice(texts_by_type[type1])
            text2 = np.random.choice(texts_by_type[type2])
            
            negative_pairs.append({
                'text1': text1,
                'text2': text2,
                'label': 0
            })
        attempts += 1
    
    print(f"Generated {len(negative_pairs)} negative pairs")
    
    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    np.random.shuffle(all_pairs)
    
    pairs_df = pd.DataFrame(all_pairs)
    print(f"Total pairs created: {len(pairs_df)}")
    print(f"Positive pairs: {len(pairs_df[pairs_df['label'] == 1])}")
    print(f"Negative pairs: {len(pairs_df[pairs_df['label'] == 0])}")
    
    # Save to CSV
    pairs_df.to_csv('pairs.csv', index=False)
    print("Saved pairs to pairs.csv")
    
    return pairs_df


def preprocess_text(text):
    """
    Preprocess text: lowercase, remove extra spaces, minimal cleaning.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def prepare_tokenizer_and_sequences(df, num_words=20000, maxlen=120, oov_token="<OOV>"):
    """
    Create tokenizer and convert texts to sequences.
    
    Args:
        df (pd.DataFrame): DataFrame with 'text1' and 'text2' columns
        num_words (int): Maximum number of words to keep
        maxlen (int): Maximum length of sequences
        oov_token (str): Token for out-of-vocabulary words
        
    Returns:
        tuple: (tokenizer, X1, X2, vocab_size)
    """
    print("\nPreparing tokenizer and sequences...")
    
    # Preprocess all texts
    texts1 = [preprocess_text(text) for text in df['text1']]
    texts2 = [preprocess_text(text) for text in df['text2']]
    
    # Combine all texts for tokenizer fitting
    all_texts = texts1 + texts2
    
    # Create and fit tokenizer
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(all_texts)
    
    # Convert texts to sequences
    sequences1 = tokenizer.texts_to_sequences(texts1)
    sequences2 = tokenizer.texts_to_sequences(texts2)
    
    # Pad sequences
    X1 = pad_sequences(sequences1, maxlen=maxlen, padding='post', truncating='post')
    X2 = pad_sequences(sequences2, maxlen=maxlen, padding='post', truncating='post')
    
    vocab_size = len(tokenizer.word_index) + 1
    if vocab_size > num_words:
        vocab_size = num_words
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Sequence shape: {X1.shape}")
    
    # Save tokenizer
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        f.write(tokenizer_json)
    print("Saved tokenizer to tokenizer.json")
    
    return tokenizer, X1, X2, vocab_size


def split_data(X1, X2, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X1 (np.ndarray): First text sequences
        X2 (np.ndarray): Second text sequences
        y (np.ndarray): Labels
        test_size (float): Proportion of test set
        val_size (float): Proportion of validation set (from remaining after test)
        random_state (int): Random seed
        
    Returns:
        tuple: (X1_train, X1_val, X1_test, X2_train, X2_val, X2_test, y_train, y_val, y_test, test_indices)
    """
    print("\nSplitting data...")
    
    # Create indices array
    indices = np.arange(len(X1))
    
    # First split: train+val and test
    indices_temp, indices_test, X1_temp, X1_test, X2_temp, X2_test, y_temp, y_test = train_test_split(
        indices, X1, X2, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train and val
    # Adjust val_size to account for test split
    val_size_adjusted = val_size / (1 - test_size)
    indices_train, indices_val, X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        indices_temp, X1_temp, X2_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"Training set: {len(X1_train)} samples")
    print(f"Validation set: {len(X1_val)} samples")
    print(f"Test set: {len(X1_test)} samples")
    print(f"Train labels - Positive: {np.sum(y_train)}, Negative: {len(y_train) - np.sum(y_train)}")
    print(f"Val labels - Positive: {np.sum(y_val)}, Negative: {len(y_val) - np.sum(y_val)}")
    print(f"Test labels - Positive: {np.sum(y_test)}, Negative: {len(y_test) - np.sum(y_test)}")
    
    return X1_train, X1_val, X1_test, X2_train, X2_val, X2_test, y_train, y_val, y_test, indices_test


def save_processed_data(X1_train, X1_val, X1_test, X2_train, X2_val, X2_test, 
                       y_train, y_val, y_test, output_dir='.'):
    """
    Save processed data as numpy arrays.
    
    Args:
        X1_train, X1_val, X1_test: First text sequences
        X2_train, X2_val, X2_test: Second text sequences
        y_train, y_val, y_test: Labels
        output_dir (str): Directory to save files
    """
    print("\nSaving processed data...")
    
    np.save(os.path.join(output_dir, 'X1_train.npy'), X1_train)
    np.save(os.path.join(output_dir, 'X1_val.npy'), X1_val)
    np.save(os.path.join(output_dir, 'X1_test.npy'), X1_test)
    np.save(os.path.join(output_dir, 'X2_train.npy'), X2_train)
    np.save(os.path.join(output_dir, 'X2_val.npy'), X2_val)
    np.save(os.path.join(output_dir, 'X2_test.npy'), X2_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print("Processed data saved as .npy files")


def main():
    """
    Main function to run all preprocessing steps.
    """
    # Load data
    df = load_data('archive')
    
    # Create pairs (limit to manageable size for training)
    # Using a reasonable number to balance dataset
    pairs_df = create_pairs(df, num_positive_pairs=50000, num_negative_pairs=50000)
    
    # Prepare tokenizer and sequences
    tokenizer, X1, X2, vocab_size = prepare_tokenizer_and_sequences(
        pairs_df, num_words=20000, maxlen=120
    )
    
    # Get labels
    y = pairs_df['label'].values
    
    # Split data
    X1_train, X1_val, X1_test, X2_train, X2_val, X2_test, y_train, y_val, y_test, test_indices = split_data(
        X1, X2, y, test_size=0.15, val_size=0.15
    )
    
    # Save processed data
    save_processed_data(X1_train, X1_val, X1_test, X2_train, X2_val, X2_test,
                       y_train, y_val, y_test)
    
    # Save test indices
    np.save('test_indices.npy', test_indices)
    
    print("\nPreprocessing complete!")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max sequence length: 120")
    
    return (X1_train, X1_val, X1_test, X2_train, X2_val, X2_test, 
            y_train, y_val, y_test, vocab_size, tokenizer, pairs_df, test_indices)


if __name__ == "__main__":
    main()

