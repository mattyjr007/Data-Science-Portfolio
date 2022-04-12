import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score, confusion_matrix


#employee data
file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/hr_data.csv"
hr_DF = pd.read_csv(file_name)

#employee statistics data
file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/employee_satisfaction_evaluation.xlsx"
emp_satis_eval = pd.read_excel(file_name)

#join the tables
main_df = hr_DF.set_index('employee_id').join(emp_satis_eval.set_index('EMPLOYEE #'))
main_df = main_df.reset_index()

#drop employee ID column
main_df.drop('employee_id',axis=1, inplace=True)


#Custom transformer to fill missing values

from sklearn.base import BaseEstimator, TransformerMixin

class missingDataTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        self.mean = X.mean()
        return self
    
    def transform(self, X, y=None):
        # Perform mean imputation on the data
        X = X.fillna(self.mean)
        return X

missingProcessor = missingDataTransformer()


#Function transformer 

from sklearn.preprocessing import FunctionTransformer

dummyProcessor = FunctionTransformer(pd.get_dummies, kw_args={"drop_first": True})

#scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# the model

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()


#Build the Pipeline

from sklearn.pipeline import Pipeline


pipeline = Pipeline([('missingProcessor',missingProcessor),
                     ('dummyProcessor',dummyProcessor),
                     ('scaler',scaler),
                     ('classifier',rf_model)])


# now split the data and fit it to the pipeline

from sklearn.model_selection import train_test_split

X = main_df.drop('left',axis=1)
y = main_df['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



#Build model model with the pipeline

pipeline.fit(X,y)

#rf_pred = pipeline.predict(X_test)
#print("Accuracy fo model {0:.2f}%".format(accuracy_score(y_test,rf_pred)*100))

#print("\n")

#print(classification_report(y_test,rf_pred))

#print("\n")

#plot_confusion_matrix(pipeline,X_test,y_test)
#plt.grid(None)

# save the model
from joblib import dump,load
dump(pipeline, 'HRR.joblib')




