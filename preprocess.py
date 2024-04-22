import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(file_path: str, test_size: float = 0.2, random_state: int = 42, split: bool = True) -> tuple:
    # Load the data
    codon_usage_df = pd.read_csv(file_path, low_memory=False)

    # Extracting the codon frequency columns and converting to numeric, coercing errors to NaN
    codon_columns = codon_usage_df.columns[5:]
    X = codon_usage_df[codon_columns].apply(pd.to_numeric, errors='coerce')
    y = codon_usage_df['Kingdom']

    # Drop rows with any NaN values in X and filter y accordingly
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]

    scaler = StandardScaler()

    if split:
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size, random_state=random_state)
    else:
        X_train, y_train = X_clean, y_clean

    # Standardize the data
    X_train_scaled = scaler.fit_transform(X_train)

    if split:
        X_test_scaled = scaler.transform(X_test)
    
    if split:
        return(X_train_scaled, X_test_scaled, y_train, y_test)
    else:
        return(X_train_scaled, y_train)
