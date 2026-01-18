# Al-for-Heart-Disease-Predication
The "Heart Disease Prediction" project focuses on predicting the presence of heart disease in individuals using machine learning techniques. By leveraging popular Python libraries such as NumPy, Pandas, and Scikit-learn (sklearn), this project provides a comprehensive solution for accurate disease prediction.

Project Overview
The "Heart Disease Prediction" project aims to develop a model that can accurately predict the presence of heart disease based on various medical factors. Early detection of heart disease is crucial for timely intervention and treatment. By employing machine learning algorithms and a curated dataset, this project offers a valuable tool for predicting heart diseas

Key Features
Data Collection and Processing: The project involves collecting a dataset containing features related to individuals' health, such as age, sex, blood pressure, cholesterol levels, and more. Using Pandas, the collected data is cleaned, preprocessed, and transformed to ensure it is suitable for analysis. The dataset is included in the repository for easy access.

Data Visualization: The project utilizes data visualization techniques to gain insights into the dataset. Matplotlib and seaborn are employed to create visualizations such as histograms, bar plots, and correlation matrices. These visualizations provide a deeper understanding of the relationships between features and help identify patterns and correlations with heart disease.

Train-Test Split: To evaluate the performance of the classification model, the project employs the train-test split technique. The dataset is divided into training and testing subsets, ensuring that the model is trained on a portion of the data and evaluated on unseen data. This allows for an accurate assessment of the model's predictive capabilities.

Classification Models: The project utilizes various classification models provided by Scikit-learn to predict the presence of heart disease. These models include logistic regression, decision trees, random forests, and support vector machines. Each model brings its own strengths and characteristics to the prediction task, enabling a comprehensive comparison of their perfor

Model Evaluation: The project evaluates the performance of the classification models using evaluation metrics such as accuracy, precision, recall, and F1 score. These metrics provide insights into the models' ability to correctly predict the presence or absence of heart disease. Additionally, visualizations such as confusion matrices are created to compare the predicted labels against the actual labels

🧠 Predicts risk of heart disease using a trained Random Forest Classifier
⚡ Simple and interactive command-line interface
📈 Uses the UCI Heart Disease dataset (Cleveland subset)
🔄 Full pipeline: model training, saving, and real-time inference
🧪 Easy to adapt for any binary classification task

🗂️ Project Structure

📂 heart-disease-predictor
 ├── 📁 data/                # Dataset folder (heart.csv)
 ├── 📁 asset/               # Contains Demo Thumbnail
 ├── 🧠 model.py             # Script to train and save the model
 ├── 🧪 evaluation.py        # Evaluation metrics and confusion matrix
 ├── 🧠 heart_disease_app.py # Command-line prediction script
 ├── 🗂️ heart_model.pkl      # Trained ML model
 ├── 📜 requirements.txt     # Python dependencies
 └── 📘 README.md            # Project overview and instructions

 📦 Model

 We use a RandomForestClassifier from scikit-learn trained on the UCI dataset.

The trained model is serialized using joblib and saved as heart_model.pkl, which can be reused without retraining.

You can use this model to make predictions with any compatible input using:

import joblib
model = joblib.load("heart_model.pkl")
prediction = model.predict([your_input])

✅ Requirements

pandas
numpy
scikit-learn
joblib
matplotlib
seaborn

📄 License

This project is not licensed.
You are free to use, modify, and share this code for personal or academic purposes.

Conclusion

The "Heart Disease Prediction" project offers a practical solution for predicting the presence of heart disease based on various medical factors. By leveraging data collection, preprocessing, visualization, and classification modeling, this project provides a comprehensive approach to addressing the prediction task. The project also includes a curated dataset to facilitate seamless exploration and experimentation.
