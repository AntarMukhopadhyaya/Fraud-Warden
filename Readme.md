# Fraud Warden
## Overview
Fraud Warden is a next-generation credit card fraud detection system that uses machine learning to predict whether a transaction is fraudulent or not. The system leverages a Random Forest Classifier to make predictions based on various features of the transaction.  
Technology Stack
Programming Language: Python
### Libraries:
- **<u>streamlit</u>** for building the web application
- **<u>pandas</u>** for data manipulation
- **<u>plotly.express</u>** and seaborn for data visualization
- **<u>scikit-learn</u>** for machine learning
- **<u>pickle</u>** for model serialization
### Installation Instructions
- Clone the Repository:  
- git clone https://github.com/yourusername/fraud-warden.git
- cd fraud-warden
- Create a Virtual Environment:
   - python -m venv venv
   - source venv/bin/activate  # On Windows use `venv\Scripts\activate`
- Install Dependencies:  
  - pip install -r requirements.txt
- Run the Application:  
   - streamlit run app.py
## How It Works
- Data Preprocessing:  
  - The application preprocesses the uploaded CSV file by removing unnecessary columns and converting date columns to datetime objects.
Additional features such as time_of_day and age are derived from existing columns.
- Feature Engineering:  
  - Categorical features are encoded into numerical values.
The data is reindexed to ensure all required columns are present.
- Oversampling:  
  - The application uses Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset.
- Model Prediction:  
  - The preprocessed data is fed into a pre-trained Random Forest Classifier model.
The model predicts whether a transaction is fraudulent based on the input features.
- Visualization:  
  - The application provides various visualizations such as histograms, bar charts, and correlation heatmaps to help users understand the data.
## Features
- **Upload CSV**: Users can upload a CSV file containing transaction data.
- **Data Preview**: Displays a preview of the uploaded data.
- **Basic Statistics**: Shows basic statistics of the dataset.
- **Data Types**: Displays the data types of each column.
- **Missing Values**: Shows the count of missing values in each column.
- **Distribution of Numerical Columns**: Visualizes the distribution of numerical columns.
- **Counts of Categorical Columns**: Visualizes the counts of categorical columns.
- **Correlation Heatmap**: Displays a heatmap of the correlation between numerical features.
- **SMOTE Sampling**: Balances the dataset using SMOTE sampling.
- **Fraud Prediction**: Predicts whether a transaction is fraudulent based on user input.
## Resources Used
- Dataset: [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- Sklearn Documentation: [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- Streamlit Documentation: [Streamlit](https://docs.streamlit.io/library)
- Plotly Documentation: [Plotly Express](https://plotly.com/python/plotly-express/)
- Seaborn Documentation: [Seaborn](https://seaborn.pydata.org/)
- Pandas Documentation: [Pandas](https://pandas.pydata.org/docs/)
- Python Documentation: [Python](https://docs.python.org/3/)
- SMOTE Documentation: [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)