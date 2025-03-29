# import numpy as np
# import pickle
#
# MODEL_PATH = "models/model.pkl"
# ENCODER_PATH = "models/label_encoder.pkl"
#
# # Load model and encoder
# with open(MODEL_PATH, "rb") as model_file:
#     model = pickle.load(model_file)
#
# with open(ENCODER_PATH, "rb") as encoder_file:
#     label_encoder = pickle.load(encoder_file)
#
#
# def predict_animal_class(animal_name):
#     """
#     Predict the classification of an animal given its name.
#     """
#     animal_name = animal_name.strip().lower()
#
#     # Check if the animal exists in the encoder
#     label_encoder_classes = [name.lower() for name in label_encoder.classes_]
#     if animal_name not in label_encoder_classes:
#         return "Unknown animal name"
#
#     # Encode input
#     encoded_input = label_encoder.transform([label_encoder.classes_[label_encoder_classes.index(animal_name)]])
#
#     # Predict class
#     prediction = model.predict([encoded_input])
#
#     # Decode predicted label
#     predicted_class = label_encoder.inverse_transform([prediction[0]])[0]
#
#     return predicted_class





import numpy as np
import pickle

MODEL_PATH = "models/model.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
NAME_ENCODER_PATH = "models/name_encoder.pkl"

# Load model and encoders
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(LABEL_ENCODER_PATH, "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

with open(NAME_ENCODER_PATH, "rb") as encoder_file:
    name_encoder = pickle.load(encoder_file)

def predict_animal_class(animal_name):
    """
    Predict the classification of an animal given its name.
    """
    animal_name = animal_name.strip().lower()

    # Check if the animal exists in the name encoder
    name_encoder_classes = [name.lower() for name in name_encoder.classes_]
    if animal_name not in name_encoder_classes:
        return "Unknown animal name"

    # Encode input using name_encoder
    encoded_input = name_encoder.transform([animal_name.title()])  # Match original casing

    # Predict class
    prediction = model.predict(encoded_input.reshape(1, -1))

    # Decode predicted label
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    return predicted_class