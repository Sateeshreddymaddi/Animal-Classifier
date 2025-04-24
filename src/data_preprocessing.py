import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_and_preprocess_data(file_path):
    """
    Processes data to use 'common_name' as feature and 'Classification' as target.
    Applies One-Hot Encoding to the feature and Label Encoding to the target.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise FileNotFoundError(f"Error reading file: {e}")

    # Validate required columns
    required_columns = {"common_name", "Classification"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset must include 'common_name' and 'Classification' columns.")

    # Clean data
    df = df[["common_name", "Classification"]].drop_duplicates().dropna()

    # Encode target (Classification)
    label_encoder = LabelEncoder()
    df["Classification"] = label_encoder.fit_transform(df["Classification"])

    # One-Hot Encode feature (common_name)
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X = onehot.fit_transform(df[["common_name"]])
    y = df["Classification"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder, onehot