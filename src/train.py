import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess_data

# Define dataset path
file_path = "../data/cleaned_animals.csv"

# Ensure models directory exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

try:
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder, onehot = load_and_preprocess_data(file_path)

    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    # Save the trained model
    with open(os.path.join(models_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)

    # Save the encoders
    with open(os.path.join(models_dir, "label_encoder.pkl"), "wb") as encoder_file:
        pickle.dump(label_encoder, encoder_file)

    with open(os.path.join(models_dir, "onehot_encoder.pkl"), "wb") as onehot_file:
        pickle.dump(onehot, onehot_file)

    print("✅ Model trained and saved successfully!")

except Exception as e:
    print(f"❌ Error: {e}")