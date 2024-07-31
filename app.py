
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
all_columns = [
    'amt', 'city_pop','category_entertainment','category_food_dining','category_gas_transport','category_grocery_net','category_grocery_pos',
    'category_health_fitness','category_home','category_kids_pets','category_misc_net','category_misc_pos','category_personal_care','category_shopping_net',
    'category_shopping_pos','category_travel',
    'gender_F','gender_M',
    'time_of_day_midday','time_of_day_morning','time_of_day_night',
    'age_Middle-Aged','age_Old','age_Young'

]


def time_of_day(hour):
    if 0 <= hour['hr_day'] <= 7:
        val = "night"
    elif 8 <= hour['hr_day'] <= 15:
        val = "morning"
    else:
        val = "midday"
    return val

def age(age):
    if 2024 - age['year_birth'] <= 29:
        val = "Young"
    elif 30 <= 2024 - age['year_birth'] <= 59:
        val = "Middle-Aged"
    else:
        val = "Old"
    return val


def preprocess_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["Unnamed: 0", "cc_num", "merchant", "first", "last", "street",
                  "lat", "long", "job", "trans_num", "unix_time", "merch_lat", "merch_long", "city", "state", "zip"],

                 axis=1)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['hr_day'] = df['trans_date_trans_time'].dt.hour
    df['time_of_day'] = df.apply(time_of_day, axis=1)
    df['year_birth'] = df["dob"].dt.year
    df['age'] = df.apply(age, axis=1)
    df = df.drop(['trans_date_trans_time', 'dob', 'hr_day', 'year_birth'], axis=1)
    return df
def preprocess_data(amount:float, gender:str, category:str, time:str, age:str,city_pop:int)->pd.DataFrame:
    if gender == "Female":
        gender = "F"
    else:
        gender = "M"
    category = category.lower().replace(" ","_")
    if time == "Morning":
        time = "morning"
    elif time == "Midday":
        time = "midday"
    else:
        time = "night"
    default_data = {col: 0 for col in all_columns}
    default_data.update({
        'amt': [amount],
        'city_pop': [city_pop],
        f'category_{category}': [1],
        'gender_F': [1 if gender == 'F' else 0],
        'gender_M': [1 if gender == 'M' else 0],
        f'time_of_day_{time}': [1],
        f'age_{age}': [1],

    })
    df = pd.DataFrame(default_data)


    # Reindex the DataFrame to ensure all columns are present
    df = df.reindex(columns=all_columns, fill_value=0)

    return df


def predict_credit_card_fraud(amount:float,gender,category,time,age,city_pop)->bool:
    X_pred = preprocess_data(amount,gender,category,time,age,city_pop)

    with open("RF-Optimized.pickle","rb") as f:
        model = pickle.load(f)
        y = model.predict(X_pred)
        print("Model Predictions: ",y)
    if y:
        return True
    else:
        return False


def main():
    st.title("Welcome to Fraud Warden")
    st.subheader("Next-Gen Credit Card Fraud Detection System")
    with st.sidebar:
        st.write("Fraud Warden is a credit card fraud detection system that uses machine learning to predict whether a transaction is fraudulent or not.")
        st.write("The system uses a Random Forest Classifier to predict whether a transaction is fraudulent or not.")
        st.write("The system uses the following features to predict whether a transaction is fraudulent or not:")
        st.write("1. Amount")
        st.write("2. City Population")
        st.write("3. Category of Transaction")
        st.write("4. Gender")
        st.write("5. Time of Transaction")
        st.write("6. Age of the Cardholder")
        st.html("<hr>")

    "Fraudulent "
    amount = st.number_input("Amount")
    city_population = st.number_input("City Population",min_value=100,max_value=1000000)
    gender = st.selectbox("Gender",["Female","Male"])
    category = st.selectbox("Category",["Entertainment","Food Dining","Gas Transport","Grocery Net","Grocery Pos","Health Fitness","Home","Kids Pets","Misc Net"])
    time = st.selectbox("Time",["Morning","Midday","Night"])
    age = st.selectbox("Age",["Young","Middle-Aged","Old"])

    uploadCsv = st.file_uploader("Upload CSV",type=['csv','xlsx'])

    if uploadCsv is not None:

        df = pd.read_csv(uploadCsv)
        df = preprocess_csv(df)
        st.write("Data Preview")
        st.write("Basic Statistics")
        st.write(df.describe())

        st.write("Data Types")
        st.write(df.dtypes)

        st.write("Missing Values")
        st.write(df.isnull().sum())

        st.write("Distribution of Numerical Columns")
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
            st.plotly_chart(fig)

        st.write("Counts of Categorical Columns")
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            counts = df[col].value_counts().reset_index()
            counts.columns = ['category', 'count']
            fig = px.bar(counts, x='category', y='count', title=f'Counts of {col}')
            st.plotly_chart(fig)


    if st.button("Predict"):
        with st.spinner("Predicting..."):
            result = predict_credit_card_fraud(amount,gender,category,time,age,city_population)

        if result:
            st.error("The following transaction might be fraudulent")
        else:
            st.success("The following transaction might not be a  fraudulent")


if __name__ == '__main__':
    main()