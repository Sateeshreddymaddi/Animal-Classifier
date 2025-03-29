import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess_data

# Define dataset path
file_path = "./data/animals_datas.csv"

# Ensure models directory exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

try:
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(file_path)

    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    with open(os.path.join(models_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)

    # Save the label encoder
    with open(os.path.join(models_dir, "label_encoder.pkl"), "wb") as encoder_file:
        pickle.dump(label_encoder, encoder_file)

    print("✅ Model trained and saved successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
