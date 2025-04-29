# 🐾 Animal Scientific Name Classifier

This project is a machine learning-based classifier that predicts the **scientific name** of an animal based on its features such as common name, classification, habitat, diet, and more. It is designed to aid in educational and zoological research tasks by providing accurate taxonomy predictions.

## 📌 Project Overview

- **Goal**: Predict the scientific name of an animal using structured data.
- **Model**: Supervised classification model using scikit-learn.
- **Dataset**: Custom dataset with fields like:
  - Common Name
  - Scientific Name
  - Classification
  - Habitat
  - Diet
  - Lifespan
  - Image Path
  - Description

## 🧠 Model Performance

- **Current Accuracy**: ~55%
- **Target Accuracy**: ≥90%
- Ongoing improvements are being made in feature engineering, data cleaning, and model optimization.

## 🗃️ Dataset

- The dataset is stored in `data/animals_dataset.csv`
- Features include both categorical and textual data
- Images are stored in the `images/` directory (if applicable)

## 📁 Project Structure

```
Animal-Classifier/
├── data/
│   └── animals_dataset.csv
├── images/
│   └── ... (animal images)
├── models/
│   └── model.pkl
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/Sateeshreddymaddi/Animal-Classifier.git
cd Animal-Classifier
```

2. **Create and activate a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app or notebook**

To explore the Jupyter notebook:

```bash
jupyter notebook notebooks/EDA_and_Modeling.ipynb
```

To run the classification app (if Flask/FastAPI is used):

```bash
python app.py
```

## 🚰 Future Improvements

- Improve model accuracy with deep learning techniques
- Include image-based classification using CNNs
- Add web interface for uploading animal data/images
- Deploy the model using Streamlit, Flask, or FastAPI
- Add scientific name pronunciation and links to Wikipedia

## 👌 Contributing

Pull requests are welcome. If you’d like to contribute, please fork the repo and use a feature branch. Issues and feedback are encouraged!

## 📄 License

This project is licensed under the MIT License.

---

*Developed with ❤️ by [Sateesh Reddy](https://github.com/Sateeshreddymaddi)*

