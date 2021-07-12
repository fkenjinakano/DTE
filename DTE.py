from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score	

class DTE:
		
	def __init__(self,
		task,
		features = True,
		output_space_features = False,
		tree_embedding_features = True,
		):
		assert(output_space_features or tree_embedding_features), "Either predictions or predictions should be true"

		self.task = task

		assert (task == "mtr" or task == "mlc"), 'The parameter (task) should be equal "mtr" for multi-target regression or "mlc" for multi-label classification'
		if task == "mtr":
			self.evaluate = self.evaluateMTR
			self.columns = ["TrainPerLayer_aRRMSE"]	
			self.bestValue = np.argmin
			# print (self.bestValue)
		elif task == "mlc":
			self.evaluate = self.evaluateMLC	
			self.columns = ["TrainPerLayer_microAUC"]
			self.bestValue = np.argmax

		self.models = []		
		self.n_folds = 3 # number of folds used for inner cross-validation procedures

		self.n_trees_rf = 150 
		self.n_trees_et = 150


		self.percentage_removal = 0.95
		
		self.pca_components = np.array([1, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95 ])

		self.sample_size = 0.5 # sample size use to generate tree-embedding features

		self.models_max_depth = 10

		self.features = features

		self.output_space_features = output_space_features
		self.tree_embedding_features = tree_embedding_features

		self.train_predictions_probabilities = []
		self.test_predictions_probabilities = []

		self.optimal_nb_components = []
		self.pcas_rf = [] 
		self.pcas_et = [] 
	
		self.model_pca_rf = []
		self.model_pca_et = []
	def fit(self, train_x, train_y):
		
		train_performance = []		
		self.mean_y = np.mean(train_y,axis=0).values
		self.models.append(self.add_layer(train_x, train_x, train_y))
		train_predictions = self.predict_proba_layer(train_x, train_x, 0)

		train_performance.append(self.evaluate(train_predictions, train_y))		
		extra_features_train_rf, extra_features_train_et  = self.get_extra_features(train_x, train_x, train_y, 0, train=True) 	
	
		if self.features:
			new_train_x_rf = pd.concat((train_x, extra_features_train_rf), ignore_index=True, axis=1)			
			new_train_x_et = pd.concat((train_x, extra_features_train_et), ignore_index=True, axis=1)			
		else:	
			new_train_x_rf = extra_features_train_rf
			new_train_x_et = extra_features_train_et
		for i in range(1, self.models_max_depth):
			self.models.append(self.add_layer(new_train_x_rf, new_train_x_et,  train_y))
			train_predictions = self.predict_proba_layer(new_train_x_rf, new_train_x_et, i)
			train_performance.append(self.evaluate(train_predictions, train_y))
	
			self.train_predictions_probabilities.append(pd.DataFrame(train_predictions))
	
			extra_features_train_rf, extra_features_train_et = self.get_extra_features(new_train_x_rf, new_train_x_et, train_y, i, train=True) 	
			
			if self.features:
				new_train_x_rf = pd.concat((train_x, extra_features_train_rf), ignore_index=True, axis=1)
				new_train_x_et = pd.concat((train_x, extra_features_train_et), ignore_index=True, axis=1)
			else:	
				new_train_x_rf = extra_features_train_rf
				new_train_x_et = extra_features_train_et
		performance = np.array(train_performance)		
	
		self.performance = pd.DataFrame(performance)
		self.performance.columns = self.columns

		self.output_nb_components = pd.DataFrame(self.optimal_nb_components, columns = ["RF_percentage","RF_components", "ET_percentage","ET_components"])
		
		self.optimal_layer = self.bestValue(train_performance[1:]) + 1 
	
	def predict_proba_optimal_layer(self):
		return self.test_predictions_probabilities[self.optimal_layer - 1]
	def add_layer(self, train_rf_x, train_et_x, train_y):
		rf = RandomForestRegressor(n_estimators=self.n_trees_rf, criterion = "mse", max_depth=None, min_samples_leaf=5, random_state=0, max_features = "sqrt", n_jobs=-1)
		et = ExtraTreesRegressor(n_estimators=self.n_trees_et, criterion = "mse", max_depth=None, min_samples_leaf=5, random_state=0, max_features = "sqrt", n_jobs=-1)	
		
		rf.fit(train_et_x, train_y)
		et.fit(train_rf_x, train_y)

		return (rf,et)
	def get_extra_features(self, x_rf, x_et, y, iteration, train):
		extra_features_et = pd.DataFrame()
		extra_features_rf = pd.DataFrame()
		if self.output_space_features:
			extra_features_predictions = self.predictions_features(x_rf, x_et, y, iteration, train)
			extra_features_predictions.columns = ["pred_layer_" + str(iteration + 1) + "_target_" + str(i + 1) for i in range(extra_features_predictions.shape[1])]
			extra_features_rf = pd.concat((extra_features_rf, extra_features_predictions), ignore_index=True, axis=1)
			extra_features_et = pd.concat((extra_features_et, extra_features_predictions), ignore_index=True, axis=1)
		
		if self.tree_embedding_features:
			extra_features_paths_rf, extra_features_paths_et = self.paths_features(x_rf, x_et, y, iteration, train)
			extra_features_paths_rf.columns = ["path_layer_rf_" + str(iteration + 1) + "_path_" + str(i + 1) for i in range(extra_features_paths_rf.shape[1])]
			extra_features_rf = pd.concat((extra_features_rf, extra_features_paths_rf), ignore_index=True, axis=1) 

			extra_features_paths_et.columns = ["path_layer_et_" + str(iteration + 1) + "_path_" + str(i + 1) for i in range(extra_features_paths_et.shape[1])]
			extra_features_et = pd.concat((extra_features_et, extra_features_paths_et), ignore_index=True, axis=1) 		

		return extra_features_rf, extra_features_et
	def predictions_features(self, x_rf, x_et, y, iteration, train): 		
		if train:
			kfold = KFold(n_splits = self.n_folds,  shuffle=True, random_state = 0)
			et_predictions = np.zeros(y.shape)
			rf_predictions = np.zeros(y.shape)
			for train_index, test_index in kfold.split(x_rf):
				rf = RandomForestRegressor(n_estimators=self.n_trees_rf, criterion = "mse", max_depth=None, min_samples_leaf=5, random_state=0, max_features = "sqrt", n_jobs=-1)
				et = ExtraTreesRegressor(n_estimators=self.n_trees_et, criterion = "mse", max_depth=None, min_samples_leaf=5, random_state=0, max_features = "sqrt", n_jobs=-1)	
				x_rf_train, y_train = x_rf.iloc[train_index], y.iloc[train_index]
				x_et_train = x_et.iloc[train_index]
				rf.fit(x_rf_train, y_train)
				et.fit(x_et_train, y_train)
			
				rf_predictions += rf.predict(x_rf)
				et_predictions += et.predict(x_et)
			predictions = (rf_predictions/self.n_folds + et_predictions/self.n_folds)/2
			return pd.DataFrame(predictions)
		else:
			return self.predict_proba_layer(x_rf, x_et, iteration)
			
	def paths_features(self, x_rf, x_et, y, iteration, train):
		if train:
			rf = RandomForestRegressor(n_estimators=self.n_trees_rf, criterion = "mse", max_depth=None, min_samples_leaf=5, random_state=0, max_features = "sqrt", n_jobs=-1)
			et = ExtraTreesRegressor(n_estimators=self.n_trees_et, criterion = "mse", max_depth=None, min_samples_leaf=5, random_state=0, max_features = "sqrt", n_jobs=-1)	
			self.model_pca_rf.append(rf)
			self.model_pca_et.append(et)
			rf_decision_path = self.decision_paths(x_rf, y, rf)
			et_decision_path = self.decision_paths(x_et, y, et)
	
			rf_transformed_features = rf_decision_path[-2]
			et_transformed_features = et_decision_path[-2]

			self.pcas_rf.append(rf_decision_path[:3]) 
			self.pcas_et.append(et_decision_path[:3])			
			self.optimal_nb_components.append((rf_decision_path[-1], rf_decision_path[2].n_components, et_decision_path[-1], et_decision_path[2].n_components))
		else:
			rf_decision_path = self.model_pca_rf[iteration].decision_path(x_rf)[0].todense()

			rf_selected_indexes = self.pcas_rf[iteration][1]

			rf_weights = self.pcas_rf[iteration][0]
			rf_pca = self.pcas_rf[iteration][2]

			et_decision_path = self.model_pca_et[iteration].decision_path(x_et)[0].todense()

			et_selected_indexes = self.pcas_et[iteration][1]
			et_weights = self.pcas_et[iteration][0]
			et_pca = self.pcas_et[iteration][2]
			
			rf_selected_nodes = rf_decision_path[:, rf_selected_indexes]
			rf_pca_input = np.multiply(rf_selected_nodes, rf_weights)
			rf_transformed_features = rf_pca.transform(rf_pca_input)

			et_selected_nodes = et_decision_path[:, et_selected_indexes]
			et_pca_input = np.multiply(et_selected_nodes, et_weights)	
			et_transformed_features = et_pca.transform(et_pca_input)
		
		rf = pd.DataFrame(rf_transformed_features)
		et = pd.DataFrame(et_transformed_features)
		return rf,et
	def decision_paths(self, train_x, train_y, model):
		kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state = 0)
		
		e = 0.001		
		n_components = self.pca_components
		best_performance = np.zeros(len(n_components)) 
		
		# valid_y = train_y.sample(frac = self.sample_size, random_state = 0)
		valid_y = train_y.sample(frac = self.sample_size, random_state = 0)

		valid_x = train_x.loc[valid_y.index]
		# print (valid_y.shape)
		for train_index, test_index in kfold.split(valid_x):	
			model_fold = clone(model)
			train_fold_x, train_fold_y = valid_x.iloc[train_index], valid_y.iloc[train_index] 
			test_fold_x, test_fold_y = valid_x.iloc[test_index], valid_y.iloc[test_index] 
			model_fold.fit(train_fold_x, train_fold_y)
			performance = np.zeros(len(n_components))					
			paths = model_fold.decision_path(train_fold_x)[0].todense()	
			summed_paths = np.sum(paths,axis=0)
			selected_indexes = np.where(summed_paths/train_fold_x.shape[0] < self.percentage_removal)
			selected_summed_nodes = summed_paths[selected_indexes]		
			selected_nodes = paths[:,selected_indexes[1]]
				
			weights = (1/(np.log(selected_summed_nodes + e )))
			pca_input = np.multiply(selected_nodes,weights)
			if pca_input.shape[1] > 0: 
				for i,c in enumerate(n_components):
					if c != 1.0:
						components = int(c * min(pca_input.shape))
					else:
						components = 1
					if components > 0:
						pca = PCA(n_components = components, random_state = 0)		

						pca.fit(pca_input)
						train_pca = pca.transform(pca_input)

						model_c = clone(model)
						model_c.fit(train_pca, train_fold_y)

						paths_test = model_fold.decision_path(test_fold_x)[0].todense()
						selected_nodes_test = paths_test[:,selected_indexes[1]]
						pca_input_test = np.multiply(selected_nodes_test,weights)

						test_pca = pca.transform(pca_input_test)
						test_pred = model_c.predict(test_pca)
						
						performance[i] = self.evaluate(test_pred, test_fold_y)
				best_performance += performance
		model.fit(valid_x, valid_y)
		paths = model.decision_path(train_x)[0].todense()	
		summed_paths = np.sum(paths,axis=0)

		selected_indexes = np.where(summed_paths/train_x.shape[0] < self.percentage_removal)					
		selected_summed_nodes = summed_paths[selected_indexes]		
		selected_nodes = paths[:,selected_indexes[1]]
		weights = (1/(np.log(selected_summed_nodes + e )))
		pca_input = np.multiply(selected_nodes,weights)		
		best_component = n_components[np.nonzero(best_performance)][self.bestValue(best_performance[np.nonzero(best_performance)])]		
		if best_component != 1: 
			components =  int(best_component * min(pca_input.shape))
		else:
			components = 1	
		pca = PCA(n_components = components, random_state = 0)			
		pca.fit(pca_input)
		return (weights, selected_indexes[1], pca, pca.transform(pca_input), best_component)

	def micro_auc(self, y_pred, y_true):
		return roc_auc_score(y_true, y_pred, average = "micro")

	def evaluateMLC(self, y_pred, y_true):	
		return self.micro_auc(y_pred, y_true)

	def evaluateMTR(self, y_pred, y_true):
		return self.aRRMSE(y_pred, y_true)
	
	def aRRMSE(self, y_pred, y_true):
		numerator = np.sum((y_true.values - y_pred) ** 2 , axis=0)
		denominator = np.sum((y_true.values - self.mean_y) ** 2 , axis=0)
		return np.mean(np.sqrt(numerator/denominator))
	def predict_proba_layer(self, x_rf, x_et, iteration):
		predictions_rf = np.array([tree.predict(x_et.astype(np.float)) for tree in self.models[iteration][0].estimators_])
		predictions_et = np.array([tree.predict(x_rf.astype(np.float)) for tree in self.models[iteration][1].estimators_])
		final_predictions = pd.DataFrame(np.mean(np.append(predictions_rf, predictions_et,axis=0),axis=0))
		return final_predictions

	def predict(self, test_x):
		
		test_performance = []
			
		test_predictions = self.predict_proba_layer(test_x, test_x, 0)
		
		extra_features_test_rf, extra_features_test_et = self.get_extra_features(test_x, test_x, None, 0, train=False)
	
		if self.features:
			new_test_x_rf = pd.concat((test_x, extra_features_test_rf), ignore_index=True, axis=1)			
			new_test_x_et = pd.concat((test_x, extra_features_test_et), ignore_index=True, axis=1)	

		else:	
			new_test_x_rf = extra_features_test_rf
			new_test_x_et = extra_features_test_et
		for i in range(1, self.optimal_layer + 1):
			test_predictions = self.predict_proba_layer(new_test_x_rf, new_test_x_et, i)
			extra_features_test_rf, extra_features_test_et  = self.get_extra_features(new_test_x_rf, new_test_x_et, None, i, train=False)
			if self.features:
				new_test_x_rf = pd.concat((test_x, extra_features_test_rf), ignore_index=True, axis=1)			
				new_test_x_et = pd.concat((test_x, extra_features_test_et), ignore_index=True, axis=1)			
			else:	
				new_test_x_rf = extra_features_test_rf
				new_test_x_et = extra_features_test_et			
		return test_predictions
