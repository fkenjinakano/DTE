from DTE import DTE
import numpy as np
import pandas as pd


"""
As described in the manuscript, our framework supports combinations of three features sets:

	1 - Tree-embeddings (TE): our proposed low dimensional decision path based feature representation

	2 - Output space (OS): output based features, a.k.a. predictions.

	3 - Input vector (X): the original feature space provided by the features in the datasets themselves.

Such combinations are arranged by initializing our model with boolean parameters, as such: 

	- df = DeepForest(features = features, prediction_features = prediction_features, path_features = path_features)

where features = X, prediction_features = OS and path_features = TE

All optimization steps, and their respective values, are defined in the DeepForest class itself and are optimized according to the aRRMSE.

For building the model, one must use the fit method, as such:
	
	- df.fit(train_x, train_y, test_x, test_y)

where, train_x and train_y correpond to the train dataset input vector, and its corresponding targets respectively. The same applies for the test dataset.

In our example, all the targets containing the key word "target".

The test dataset is included in the fit method because all outputs are calculated during the training. 

The outcomes of the model are accessed directly via its object. We list them below:

	1 - train_prediction_probabilities: A list containing the output of each layer using the training dataset. 
		- df.train_predictions_probabilities 
	2 - output_nb_components: A list containing the tuples representing the optimal number of components and its corresponding percentage at each layer. 		
	The first position correpsonds to the Random Forest, and the second for the Extra-trees. 
		- df.output_nb_components
	3 - performance: A list containing the aRRMSE from both training and testing dataset obtained at each layer. 
		- df.performance
	4 - optimal_layer = A integer that represents the layer selected as the optimal w.r.t to the stopping criterion (best performance in the training dataset). 
		- df.optimal_layer
	5 - predict_proba_optimal_layer = A method that returns the predictions obtained in the optimal layer using the test dataset, a.k.a. test_prediction_probabilities[optimal_layer]
	Useful for using our method as a comparison. 
		- df.predict_proba_optimal_layer()

"""


features = True
prediction_features = False
path_features = True

train = pd.read_csv("exampleDatasetMTR/train.csv")
test = pd.read_csv("exampleDatasetMTR/test.csv")
		
train_x = train.drop([t for t in train.columns if "target"  in t],axis=1)
train_y = train[[t for t in train.columns if "target"  in t]]

test_x = test.drop([t for t in test.columns if "target"  in t],axis=1)
test_y = test[[t for t in test.columns if "target"  in t]]
	
df = DTE(task = "mtr", features = features, prediction_features = prediction_features, path_features = path_features)
df.fit(train_x, train_y)
predictions = df.predict(test_x)


# print (df.performance)
# print (df.train_predictions_probabilities)
# print (df.output_nb_components)
# print (df.optimal_layer)
