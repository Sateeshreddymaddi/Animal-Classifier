import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess_data

# Paths
file_path = "./data/animals_datas.csv"
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

try:
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder, name_encoder = load_and_preprocess_data(file_path)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and encoders
    with open(os.path.join(models_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(models_dir, "name_encoder.pkl"), "wb") as f:
        pickle.dump(name_encoder, f)

    print("✅ Model trained and saved successfully!")

except Exception as e:
    print(f"❌ Error: {e}")