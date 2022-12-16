# xgboost model_predictor
It is a forward formulation and computation as used to predict xgb score. 
Inputs: 1) tabular feature data in csv file;
2) xgboost model file in .txt format which is produced by using Scikit-learn library.
Output:
xgboost score for each object (row of the csv file)

Code flow chart:
Build Trees: read in the model file and construct it into a list of multi-trees. Each element of the list is the root note of a single tree.
Read feature data which is stored in csv file format.
Compute Score: loop through the rows of the feature data file, and compute the xgb score for each instance of row.

Time complexity: n_tree x log(n_node) X n_sample, n_tree: number of trees, n_node: node number of a single tree, n_sample: number of instances.
