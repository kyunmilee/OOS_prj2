import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np

def sort_dataset(dataset_df):
	def1_result = dataset_df.sort_values(by='year', ascending=True)
	return def1_result

def split_dataset(dataset_df):
	dataset_df['salary'] = dataset_df['salary'] * 0.001
	
	x = dataset_df.drop('salary', axis=1)
	y = dataset_df['salary']
	
	x_train = x.iloc[:1718]
	x_test = x.iloc[1718:]
	y_train = y.iloc[:1718]
	y_test = y.iloc[1718:]
	
	return x_train, x_test, y_train, y_test

def extract_numerical_cols(dataset_df):
	numerical_columns = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
	numerical_df = dataset_df[numerical_columns]
	
	return numerical_df

def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_re = DecisionTreeRegressor()
    dt_re.fit(X_train, Y_train)
    dt_predict = dt_re.predict(X_test)
    
    return dt_predict

def train_predict_random_forest(X_train, Y_train, X_test):
	rf_re = RandomForestRegressor()
	rf_re.fit(X_train, Y_train)
	rf_predict = rf_re.predict(X_test)
 
	return rf_predict

def train_predict_svm(X_train, Y_train, X_test):
	svm_pipe = make_pipeline(
		StandardScaler(),
		SVR()
	)
	svm_pipe.fit(X_train, Y_train)
	svm_pipe_predict = svm_pipe.predict(X_test)
 
	return svm_pipe_predict
 

def calculate_RMSE(labels, predictions):
	rmse_val = np.sqrt(np.mean((labels-predictions)**2))
 
	return rmse_val

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))