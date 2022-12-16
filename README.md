# xgboost model_predictor

It is a forward formulation for predicting xgboost score given inputs of a trained model files in .txt and a feature data file in .csv. 

Inputs: 
  1) tabular feature data file in .csv format (rows: instances, columns: feature data including header of feature names);
  2) xgboost model file in .txt format which is generated from the Scikit-learn library of xgb.train ().
  
Output:
xgboost scores for all instances (rows of the csv file), one score for one row (a feature vector).

Code flow chart:
Build trees: read in the txt model file and construct it into a vector of multi-trees. Each element of the vector is the root note of a single binary-tree.
Read feature data which is stored in csv file format.
Compute xgboost score: loop through the rows of the feature data file, and compute a xgb score for each row (a vector of features).

Time complexity: n_tree x log(n_node) x n_instance
Space complexity: n_tree x n_node x n_instance
n_tree: number of trees, n_node: node number of a single tree (it varies for different tree), n_sample: number of instances.

