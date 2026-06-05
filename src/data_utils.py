# src/data_utils.py

import pandas as pd


def load_gervader_words(filepath, threshold):
    """
    Load words from the GerVADER lexicon based on a sentiment threshold.

    The function reads a tab-separated file containing words and their sentiment scores.
    Only words whose absolute score meets or exceeds the given threshold are included.

    Parameters
    ----------
    filepath : str
        Path to the GerVADER lexicon file.

    threshold : float
        Minimum absolute sentiment score required to include a word.

    Returns
    -------
    list of str
        List of filtered words in lowercase.
    """

    words = []

    # Open and read lexicon file
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")

            # Skip malformed lines
            if len(parts) < 2:
                continue

            word = parts[0].lower()
            score = float(parts[1])

            # Include word if it meets threshold
            if abs(score) >= threshold:
                words.append(word)

    return words


def load_vader_words(analyzer, threshold=1.5):
    """
    Extract words from a VADER sentiment analyzer lexicon based on a threshold.

    The function filters words by their sentiment polarity score provided by VADER.

    Parameters
    ----------
    analyzer : SentimentIntensityAnalyzer
        Initialized VADER sentiment analyzer instance.

    threshold : float, optional (default=1.5)
        Minimum absolute sentiment score required to include a word.

    Returns
    -------
    list of str
        List of filtered words in lowercase.
    """

    words = []

    # Iterate over VADER lexicon entries
    for word, score in analyzer.lexicon.items():

        # Include word if threshold is met
        if abs(score) >= threshold:
            words.append(word.lower())

    return words


def remove_duplicate_texts(df, text_col="document"):
    """
    Remove duplicate text entries from a DataFrame.

    The function normalizes text (lowercase + strip) and removes duplicate rows
    based on that normalized version.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing text data.

    text_col : str, optional (default="document")
        Column name containing the text.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicate texts removed.
    """

    # Ensure the column exists
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")

    # Size before deduplication
    before = len(df)

    # Normalize text (important for consistent duplicate detection)
    df["_norm_text"] = (
        df[text_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Drop duplicates based on normalized text
    df_clean = df.drop_duplicates(subset="_norm_text").copy()

    # Remove helper column
    df_clean = df_clean.drop(columns=["_norm_text"])

    # Size after deduplication
    after = len(df_clean)

    # Print summary statistics
    print("\n=== Duplicate Removal ===")
    print(f"Before: {before} rows")
    print(f"After:  {after} rows")
    print(f"Removed: {before - after} duplicates")

    return df_clean