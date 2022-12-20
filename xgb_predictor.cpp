// XGBoost model predictor// 
// Author: Qinwu Xu
// Date: 01-21-2022
// version 1.0

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <unordered_map>
#include<string.h>
using namespace std;

//tree node as a binary search  tree
class Node {
    //initilization for each node
    int left_id = -1;
    int right_id = -1;
    int missing_id = -1;
    string feature = "None";
    double fea_threshold = 0;
    double leaf_val = 0;
public:
    //tree node
    Node(int left_val, int right_val, int missing_val, string feature_, double fea_threshold_) {
        left_id = left_val;
        right_id = right_val;
        feature = feature_;
        fea_threshold = fea_threshold_;
        missing_id = missing_val;
    }
    // leaf node
    Node(double leaf_val_) {
        leaf_val = leaf_val_;
        feature = "leaf";
    }
};

//read in model file and build trees into a vector storing unordered_map (key: node ID, value: tree node)
class Tree {
public:
    string xgb_model;
    Tree (string model_file) {
        xgb_model = model_file;
    }

    vector <unordered_map<int, Node> > read_tree_model() {
        fstream newfile;
        //ifstream newfile;
        vector <unordered_map<int, Node>> trees;
        std::string line;
        
        unordered_map <int, Node> nodes_map;
        //unordered_map <int, double> test;
        newfile.open(xgb_model, ios::in);
        if (newfile.is_open()) {
            while (std::getline(newfile, line))
            {
                vector <string> wordVector;
                std::size_t prev = 0, pos;
                while ((pos = line.find_first_of(":[<] =,", prev)) != std::string::npos)
                {
                    if (pos > prev)
                        wordVector.push_back(line.substr(prev, pos - prev));
                    prev = pos + 1;
                }

                if (prev < line.length())
                    wordVector.push_back(line.substr(prev, std::string::npos));

                if (wordVector[0] == "booster") {
                    if (nodes_map.size() >= 1) {
                        trees.push_back(nodes_map);
                        nodes_map.clear();
                    }
                }
                else {
                    int id = stoi(wordVector[0]);
                    if (wordVector[1] == "leaf")
                        nodes_map.emplace(id, Node(stod(wordVector[2]))); //Node(int node_id, double leaf_val)
                    else
                        nodes_map.emplace(id, Node(stoi(wordVector[4]), stoi(wordVector[6]), stoi(wordVector[8]), wordVector[1], stod(wordVector[2])));
                }
            }
        }
        if (nodes_map.size() >= 1) { trees.push_back(nodes_map); }
        return trees;
    }
};

// open feature data csv file and read in row by row
vector <unordered_map <string, double>> read_table_input_data(string in_file)
{
    const char* feature_data_file = in_file.c_str();
    ifstream fin (feature_data_file);
    vector<string> feature_names;
    vector<unordered_map <string, double>> feature_values;    
    vector<string> row;
    string word, temp;
    string line;
    unordered_map <string, double > obj_fea_values;

    if (fin.is_open()) {
        //ifstream fin (in_file);
        getline(fin, line);
        stringstream s0(line);
        while (getline(s0, word, ',')) {
            feature_names.push_back(word);
        }
        feature_names[0] = "Object Number";
        
        while (getline(fin, line)) {
            obj_fea_values.clear();
            stringstream s(line);
            int i = -1;
            while (getline(s, word, ',')) {
                i++;
                obj_fea_values[feature_names[i]] = stod(word);
            }
            feature_values.push_back(obj_fea_values);
        }
    }
    return feature_values;
}

//xgb score predictor
class xgb_predictor {
public:
    //compute score for each single tree
    double score_single_tree(unordered_map <int, Node>& single_tree, Node* root, unordered_map <string, string>& map_features, unordered_map <string, double>& features) {
        string feature = root->feature;
        if (feature == "leaf") {
            return root->leaf_val;
        }
        double fea_value = 0;
        if (map_features.find(feature) != map_features.end())
            feature = map_features[feature]; // warning C4244 : 'argument' : conversion from 'double' to 'const _Elem', possible loss of data

        if (features.find(feature) != features.end()) {
            fea_value = features[feature];
            if (fea_value < root->fea_threshold)
                return score_single_tree(single_tree, &single_tree.at(root->left_id), map_features, features);
            else
                return score_single_tree(single_tree, &single_tree.at(root->right_id), map_features, features);
        }
        else {
            return score_single_tree(single_tree, &single_tree.at(root->missing_id), map_features, features);
        }
}

// compute the total score accumulated from every single tree
double score(vector< unordered_map <int, Node>>& trees, unordered_map <string, string>& map_features, unordered_map <string, double>& features) {
    double xgb_score = 0.5;
    for (unordered_map <int, Node> single_tree: trees) {
        double single_score = score_single_tree(single_tree, &single_tree.at(0), map_features, features);
        xgb_score += single_score;// score_single_tree(single_tree, & single_tree.at(1), map_features, features);
    }
    return xgb_score;
}
};

int main ()
{
    //feature name mapping between xgboost model and feature data file (xgboost model can not read in feature name/string with space between. Here I use "_" to replace spaces of the original feature names)
    unordered_map <string, string> map_features;
    // user needs to customize the feature names below
    map_features = {{"ab_ab", "ab ab"}, 
                    {"cd_cd", "cd cd"}, 
                    {"ef_ef", "ef ef"}
                    };
    unordered_map <string, double> obj_feature_values;
    string in_file = "xgb_model_BULK.txt"; //model file generated by Sciki-learn library from Python ML training
    string feature_data_file = "Sample.csv";
    vector <unordered_map <string, double>> feature_values = read_table_input_data (feature_data_file);

    Tree* xgb_tree = new Tree(in_file);
    vector <unordered_map <int, Node>> trees = xgb_tree->read_tree_model();
    xgb_predictor predictor = xgb_predictor();
    vector <double> xgb_scores;
    ofstream myfile;
    myfile.open("xgb_scores.txt");
    for (auto obj_features : feature_values) {
        double score = predictor.score(trees, map_features, obj_features);
        xgb_scores.push_back(score);
        myfile << score << endl;
    }
    myfile.close();
    delete xgb_tree;
    return 0;
}


