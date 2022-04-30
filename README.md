# xgb_predict
It is used to predict xgb score of binary classification problem. Given the trained tree information, it computes the probability of class 1
The input decision tree follows the format from the genration of xgb.train of skit-learn library, Not the xgb.fit which (the later) instead computes the probability for both classes of 0 and 1
Flow chart:
class Tree: Read the decision tree information -> cached into a list of tree root nodes (each nodel represents the root of a tree)-> build into a full binary tree
Class probability: caclulate the probability of each tree and accumulate for a single instance.
