from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import os

app = Flask(__name__)

# Load the dataset
DATA_PATH = "data/animals_data.csv"
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()  # Remove extra spaces in column names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    animal_name = request.form.get('animal_name', '').strip().lower()

    # Search for the animal in the dataset (case insensitive)
    result = df[df["common_name"].str.lower() == animal_name]

    if not result.empty:
        animal_details = result.iloc[0].to_dict()
        return render_template('index.html', animal_details=animal_details)
    else:
        error_message = f"No data found for '{animal_name}'. Please try another animal."
        return render_template('index.html', error=error_message)

# Serve static images correctly
@app.route('/static/images/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static/images'), filename)

if __name__ == '__main__':
    app.run(debug=True)
