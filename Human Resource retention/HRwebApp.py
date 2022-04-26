from joblib import load
from missDT import missingDataTransformer
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

def get_model(filename):

    model = load(filename=filename)
    return model


def encode_Yes(inpt):

    if inpt == 'Yes':
        return 1
    else:
        return 0


main_df = get_data()
model = get_model('HRR.joblib')


header = st.container()
dataset = st.container()
visualization = st.container()
prediction = st.container()


with st.sidebar:
    st.header('Select Input data to predict')

    no_proj = st.slider('Number of project by employee?',min_value=main_df['number_project'].min()-1,max_value=main_df['number_project'].max() +3,value=2,step=1)

    avr_mothly_hrs = st.slider('Average monthly working hours?',min_value=main_df['average_montly_hours'].min()-26,max_value=main_df['average_montly_hours'].max() + 10,value=120,step=1)

    yrs_spent = st.slider('Years spent in company?',min_value=main_df['time_spend_company'].min()-1,max_value=main_df['time_spend_company'].max() +5,value=2,step=1)

    work_accident = st.selectbox('Ever had work accident ?',('Yes','No'))
    work_acciden_ = encode_Yes(work_accident)

    Promoted_inpast_5yrs = st.selectbox('Been promoted in the last 5yrs?',('Yes','No'))
    Promoted_inpast_5yr_ = encode_Yes(Promoted_inpast_5yrs)

    department = st.selectbox('Department?',tuple(main_df.department.unique()))

    salary = st.selectbox('Salary type',tuple(main_df.salary.unique()))

    satisfactory_level = st.number_input('Satisfactory level?',min_value=0.0, max_value=main_df['satisfaction_level'].max())

    last_evaluation  = st.number_input('Last evaluation?',min_value=0.0, max_value=main_df['last_evaluation'].max())

    predict = st.button('Predict')


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
    fig1_col.write('numbers of staff in each department.')

    fig2_col.plotly_chart(fig2,use_container_width=True)
    fig2_col.write('numbers of staff by category of salary they earn.')



with prediction:

    st.header("Predition from the inputed variable")
    if predict:

        value = model.predict([[no_proj,avr_mothly_hrs,yrs_spent, work_acciden_,Promoted_inpast_5yr_,department,salary,satisfactory_level,last_evaluation]])
        
        st.write("The predicted value from the input param is", value[0])

        if value[0] == 0:
            st.write('this employee is likely to leave.')
        else:
            st.write('this employee is likely to stay.')    
        

    else:
        st.write('waiting for input param....')    
