# xgboost model_predictor

It is a forward formulation for predicting xgboost score given inputs of a trained model files in .txt and a feature data file in .csv. 

Inputs: 
  1) tabular feature data file in .csv format (rows: instances, columns: feature data including header of feature names);
  2) xgboost model file in .txt format which is generated from the Scikit-learn library of xgb.train ().
  
Output:
xgboost scores for all instances (rows of the csv file), one score for one row (a feature vector).

Code flow chart:
1) Build trees: a) read in the model txt file and construct it into a vector of multi-trees. b) built every single binary tree, and store it as an unordered_map as the element of the vector. The unordered_map includes the node IDs as key and the binary-tree nodes as its value;
2) Open feature data csv file and streaming row by row (one row is a vector of feature values).
3) Compute xgboost score: loop through the rows of the feature data file, and accumulate the computation of a total xgb score from rows.

Time complexity: n_tree x log(n_node) x n_instance.
Space complexity: n_tree x n_node x n_instance.
where, 
n_tree - number of trees, 
n_node - node number of a single tree (it varies for different tree), 
n_sample - number of instances.

