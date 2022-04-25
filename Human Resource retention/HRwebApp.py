from joblib import load
from pip import main
import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache
def get_data():
    #employee data
    file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/hr_data.csv"
    hr_DF = pd.read_csv(file_name)

    #employee statistics data
    file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/employee_satisfaction_evaluation.xlsx"
    emp_satis_eval = pd.read_excel(file_name)

    #join the tables
    main_df = hr_DF.set_index('employee_id').join(emp_satis_eval.set_index('EMPLOYEE #'))
    main_df = main_df.reset_index()

    return main_df


main_df = get_data()

header = st.container()
dataset = st.container()
visualization = st.container()
prediction = st.container()


with header:
    st.title('Human resource retention App')
    st.image("https://www.insperity.com/wp-content/uploads/employee-retention-strategies_640x302.jpg", caption="Source: insperity")
    st.write("This web app is built to determine the retention of staffs based on some notable features made available by the Human resource.")


with dataset:
    st.header("HR dataset")
    st.write("random sampled data of 10 staffs")
    st.dataframe(main_df.sample(10))



with visualization:
    st.header("Exploratory analysis")
    fig1 = px.histogram(main_df,x='department',text_auto=True).update_xaxes(categoryorder="total descending")
    fig2 = px.histogram(main_df,x='salary',text_auto=True).update_xaxes(categoryorder="total descending")
    
    fig1_col, fig2_col = st.columns(2)

    fig1_col.plotly_chart(fig1,use_container_width=True)

    fig2_col.plotly_chart(fig2,use_container_width=True)

