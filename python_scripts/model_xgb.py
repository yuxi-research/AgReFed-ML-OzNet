"""
Extreme Gradient Boosting Model.
Mannual: https://www.kaggle.com/code/stuarthallows/using-xgboost-with-scikit-learn
		 https://xgboost.readthedocs.io/en/stable/python/python_api.html

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from xgboost import XGBRegressor


print_info = False


def xgb_train(X_train, y_train, tune=False):
	"""
	Trains XGB regression model with trainig data

	INPUT
	X: input data matrix with shape (npoints,nfeatures)
	y: target varable with shape (npoints)

	RETURN
	xgb_model: trained xgb model
	"""

	if tune:
		# Grid Search CV
		xgb_reg = XGBRegressor(random_state=42)
		print('Starting Grid Search CV for XGBoosting')
		# param_grid = {'n_estimators': [100, 200, 300],
		# 		# 'max_depth': [None, 10, 20], 
		# 		# 'max_leaves': [0, 1, 5, 10],
		# 		# 'min_samples_split': [2, 5, 10],
		# 		# 'min_samples_leaf': [1, 2, 4],
		# 		# 'max_features': [0.3, 0.5, 0.7, 0.9],
		# 		}

		param_grid = {
			'n_estimators': [100, 200, 300],
			'max_depth': [None, 10, 20],
			'learning_rate': [0.1, 0.01, 0.001],
			'subsample': [0.8, 0.9, 1.0],
			'colsample_bytree': [0.8, 0.9, 1.0],
			'reg_alpha': [0, 0.1, 0.5],
			'reg_lambda': [0, 0.1, 0.5],
			'gamma': [0, 0.1, 0.5]
		}
		grid_search = GridSearchCV(xgb_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=5)
		grid_search.fit(X_train, y_train)
		best_params = grid_search.best_params_
		print('Best Parameters', grid_search.best_params_)
		xgb_reg = XGBRegressor(**best_params, random_state = 42)
	else:
		print("Using default hyperparameters for XGBoosting")
		
		param_optimal = {'n_estimators': 1000, 
						'colsample_bytree': 0.8, 
						'gamma': 0, 
						'learning_rate': 0.1, 
						'max_depth': None, 
						'max_leaves': 0, 
						'reg_alpha': 0, 
						'reg_lambda': 0, 
						'subsample': 0.8}

		print(param_optimal)

		xgb_reg = XGBRegressor(**param_optimal, random_state=42) # 'reg:linear' depreceasted 

	xgb_reg.fit(X_train, y_train)
	return xgb_reg


def xgb_predict(X_test, xgb_model, y_test = None, outpath = None):
	"""
	Returns Prediction for Random Forest regression model

	INPUT
	X_text: input datpoints in shape (ndata,n_feature). THe number of features has to be the same as for the training data
	xgb_model: pre-trained Extreme Gradient Boosting regression model
	y_test: if True, uses true y data for normalized RMSE calculation

	Return
	ypred: predicted y values
	ypred_std: standard deviation of prediction
	rmse_test: RMSE of test data (if y_test is not None)
	"""
	ypred = xgb_model.predict(X_test)
	ypred_std = 1e-6 * np.ones(ypred.shape)

	if y_test is not None:
		# calculate RMSE
		rmse_test = np.sqrt(np.mean((ypred - y_test)**2)) / y_test.std()
		if print_info: print("Extreme Gradient Boosting normalized RMSE Test: ", np.round(rmse_test, 4))
		if outpath is not None:
			plt.figure()  # inches
			plt.title('Extreme Gradient Boosting Test Data')
			# plt.errorbar(y_test, ypred, ypred_std, linestyle='None', marker = 'o', c = 'b')
			plt.scatter(y_test, ypred, c = 'b')
			plt.xlabel('y True')
			plt.ylabel('y Predict')
			plt.savefig(os.path.join(outpath, 'xgb_test_pred_vs_true.png'), dpi = 300)
			plt.close('all')
	else:
		rmse_test = None

	return ypred, ypred_std, rmse_test


def xgb_train_predict(X_train, y_train, X_test, y_test = None, outpath = None):
	"""
	Trains Random Forest regression model with trainig data and returns prediction for test data

	INPUT
	X_train: input data matrix with shape (npoints,nfeatures)
	y_train: target varable with shape (npoints)
	X_test: input data matrix with shape (npoints_test,nfeatures)
	y_test: target varable with shape (npoints_test)
	outpath: path to save plots

	RETURN
	ypred: predicted y values
	residuals: residuals of prediction
	"""

	# Train XGB
	xgb_model = xgb_train(X_train, y_train)

	# Predict for X_test
	ypred, ypred_std, nrmse_test = xgb_predict(X_test, xgb_model, y_test = y_test, outpath = outpath)

	# calculate square errors
	if y_test is not None:
		residual = ypred - y_test
	else:
		residual = np.zeros_like(y_test)
	return ypred, residual, xgb_model


def test_xgb(logspace = False, nsamples = 600, nfeatures = 14, ninformative = 12, noise = 0.2, outpath = None):
	"""
	Test XGB model on synthetic data

	INPUT
	logspace: if True, uses logarithmic scale for features
	nsamples: number of samples
	nfeatures: number of features
	ninformative: number of informative features
	noise: noise level
	outpath: path to save plots

	"""
	# Create simulated data
	from sklearn.datasets import make_regression
	Xorig, yorig, coeffs = make_regression(n_samples=nsamples, 
		n_features=nfeatures, n_informative=ninformative, 
		n_targets=1, bias=2.0, tail_strength=0.2, noise=noise, shuffle=True, coef=True, random_state=42)
	if logspace:
		Xorig = np.exp(Xorig)
		yorig = np.exp(yorig/100)

	if outpath is not None:	
		os.makedirs(outpath, exist_ok = True)

	X_train, X_test, y_train, y_test = train_test_split(Xorig, yorig, test_size=0.5, random_state=42)

	# Run XGB
	y_pred, residual, xgb_model = xgb_train_predict(X_train, y_train, X_test, y_test = y_test, outpath = outpath)

	# Calculate normalized RMSE:
	nrmse = np.sqrt(np.nanmean(residual**2)) / y_test.std()
	nrmedse = np.sqrt(np.median(residual**2)) / y_test.std()
	if print_info:
		print('Normalized RMSE for test data: ', np.round(nrmse,3))
		print('Normalized ROOT MEDIAM SE for test data: ', np.round(nrmedse,3))
	#Feature Importance
	if print_info:
		print('Model correlation coefficients:', coeffs )

