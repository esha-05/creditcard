Credit Card Score Prediction Using Machine Learning
Project Overview
This project aims to predict the credit scores of individuals using machine learning techniques based on their financial data, such as income, credit history, and debt-to-income ratio. The system uses decision trees, random forests, and convolutional neural networks (CNN) to classify users' creditworthiness for future loans based on their historical data.

Features
Multiple ML Models: Includes decision tree, random forest, and CNN models for predicting credit scores.
Data Preprocessing: Cleans and preprocesses the financial dataset for training.
User Interface: Provides an intuitive UI to load data, select models, and display results.
Performance Metrics: Displays the accuracy of each model, allowing users to compare and select the best-performing one.
Motivation
Assessing an individual's creditworthiness accurately helps financial institutions mitigate risks and ensures responsible lending practices. Machine learning provides an efficient way to automate this credit scoring process and enhance decision-making.

Problem Statement
To build a system that automates the evaluation of credit applications, predicts users' credit scores, and assesses their likelihood of defaulting on loans using machine learning techniques.

Models and Algorithms
Decision Tree Classifier: A simple flowchart-like tree structure for classification.
Random Forest Classifier: A robust model that reduces overfitting by aggregating multiple decision trees.
Convolutional Neural Network (CNN): A deep learning model used for advanced predictions.
Prerequisites
The following libraries are required to run the project:

Python 3.6+
Flask
Pandas
Numpy
Scikit-learn
TensorFlow (for CNN)
To install dependencies, use:

bash
Copy code
pip install -r requirements.txt
Dataset
The dataset contains information about individuals' financial history, including:

Age
Income
Credit history (years)
Debt-to-income ratio
Employment history

File Structure
/templates: HTML files for user interface.
/static: Static assets such as CSS and images.
app.py: Main Flask application.
models.py: Contains the machine learning models and their training scripts.
requirements.txt: List of dependencies.

How to Run
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/credit-card-score-prediction.git
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Run the Flask server:
bash
Copy code
python app.py
Open the browser and navigate to http://127.0.0.1:5000/ to load the application.

Future Enhancements
Real-time data updates: Implement real-time updates for predictions using live financial data.
Advanced deep learning models: Experiment with more complex models for improved accuracy.

Contributors
Ritesh Singh
Karthik V
Esha T
Arjun V

License
This project is licensed under the MIT License.

