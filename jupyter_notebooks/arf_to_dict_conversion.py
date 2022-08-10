import numpy as np
from river.tree.nodes.branch import NumericBinaryBranch
from river.tree.nodes.leaf import HTLeaf

def extract_rg_arf(arf_model, feature_names):
    """
    Extract tree weights from the Adaptive Random Forest Regressor implemented by River.
    
    Parameters
    ----------
    arf_model: river.ensemble.AdaptiveRandomForestRegressor
        The adaptive random forest regressor object.
    feature_names: list of str
        List containing the feature names of offline/online train set.
        
    Returns
    -------
    arf_dict : dict
        The dictionary containing the extracted tree weights.
    """
    features_idx = {feature: idx for idx, feature in enumerate(feature_names)}
    N_OUTPUTS = 1
    arf_dict = {
        "internal_dtype": np.float64,
        "input_dtype"   : np.float64,
        "objective"     : "squared_error",
        "tree_output"   : "raw_value",
        "base_offset"   : 0,
        "trees"         : []
    }
    total_models = len(arf_model)
    scaling = 1.0 / total_models

    for hoeffding_tree_member in arf_model:
        hoeffding_tree = hoeffding_tree_member.model
        total_nodes = hoeffding_tree.n_nodes
        node_count = 0
        
        hoeffding_dict = {
            "children_left"     : np.empty(total_nodes),
            "children_right"    : np.empty(total_nodes),
            "features"          : np.empty(total_nodes),
            "thresholds"        : np.empty(total_nodes, dtype = np.float64),
            "node_sample_weight": np.empty(total_nodes, dtype = np.float64),
            "values"            : np.empty((total_nodes, N_OUTPUTS))
        }
        
        # Each queue elements: (current node, current node index, boolean index array)
        # New element is appended when a new node can be further splitted
        queue = [(hoeffding_tree._root, 0)]

        while len(queue) > 0:
            node, node_idx = queue.pop(0)
            flip = False
            
            if isinstance(node, HTLeaf):
                # Assign -1 since the current node is a leaf node
                hoeffding_dict["children_left"][node_idx] = -1
                hoeffding_dict["children_right"][node_idx] = -1
                # Assign -2 since the current node is a leaf node
                hoeffding_dict["features"][node_idx] = -2
                hoeffding_dict["thresholds"][node_idx] = -2
            else:
                feature = node.feature
                
                # Get the split value and the filters for both left and right child
                if isinstance(node, NumericBinaryBranch):
                    split_threshold = node.threshold
                else: # isinstance(node, NominalBinaryBranch)                    
                    split_threshold = node.value
                    # Check if the left and right child need to be flipped
                    if split_threshold == 1.0:
                        flip = True
                    split_threshold = 0.5
            
                # Update the feature and split value of the current node
                hoeffding_dict["features"][node_idx] = features_idx[feature]
                hoeffding_dict["thresholds"][node_idx] = split_threshold
            
                # Add the child nodes to the queue
                if flip:
                    node_count += 1
                    hoeffding_dict["children_left"][node_idx] = node_count
                    queue.append((node.children[1], node_count))
                    node_count += 1
                    hoeffding_dict["children_right"][node_idx] = node_count
                    queue.append((node.children[0], node_count))
                
                else:
                    node_count += 1
                    hoeffding_dict["children_left"][node_idx] = node_count
                    queue.append((node.children[0], node_count))
                    node_count += 1
                    hoeffding_dict["children_right"][node_idx] = node_count
                    queue.append((node.children[1], node_count))

            # Retrieve and scale down the prediction value by the total number of base learners
            hoeffding_dict["values"][node_idx] = node.stats.mean.get() * scaling

            """
            In order to prevent the error shown below, the value must not be zero when the dictionary is ingested into tree SHAP explainer using 'tree_path_dependent' approach. 

            AssertionError: The background dataset you provided does not cover all the leaves in the model, so TreeExplainer cannot run with the feature_perturbation="tree_path_dependent" option! Try providing a larger background dataset, no background dataset, or using feature_perturbation="interventional".
            """
            # Retrieve the weighted count of samples at the current node
            node_sample_weight = 1 if node.stats.mean.n == 0 else node.stats.mean.n
            hoeffding_dict["node_sample_weight"][node_idx] = node_sample_weight

        hoeffding_dict["children_default"] = hoeffding_dict["children_left"]
        arf_dict["trees"].append(hoeffding_dict)

    return arf_dict

def extract_cf_arf(arf_model, feature_names):
    """
    Extract tree weights from the Adaptive Random Forest Classifier implemented by River.
    
    Parameters
    ----------
    arf_model: river.ensemble.AdaptiveRandomForestClassifier
        The adaptive random forest classifier object.
    feature_names: list of str
        List containing the feature names of offline/online train set.
        
    Returns
    -------
    arf_dict : dict
        The dictionary containing the extracted tree weights.
    """
    features_idx = {feature: idx for idx, feature in enumerate(feature_names)}
    N_OUTPUTS = 1
    NUM_CLASSES = 2
    arf_dict = {
        "internal_dtype": np.float64,
        "input_dtype"   : np.float64,
        "objective"     : "binary_crossentropy",
        "tree_output"   : "probability",
        "base_offset"   : 0,
        "trees"         : []
    }
    total_models = len(arf_model)
    scaling = 1.0 / total_models

    for hoeffding_tree_member in arf_model:
        hoeffding_tree = hoeffding_tree_member.model
        total_nodes = hoeffding_tree.n_nodes
        node_count = 0
        
        hoeffding_dict = {
            "children_left"     : np.empty(total_nodes),
            "children_right"    : np.empty(total_nodes),
            "features"          : np.empty(total_nodes),
            "thresholds"        : np.empty(total_nodes, dtype = np.float64),
            "node_sample_weight": np.empty(total_nodes, dtype = np.float64),
            "values"            : np.empty((total_nodes, N_OUTPUTS, NUM_CLASSES))
        }
        
        # Each queue elements: (current node, current node index, boolean index array)
        # New element is appended when a new node can be further splitted
        queue = [(hoeffding_tree._root, 0)]

        while len(queue) > 0:
            node, node_idx = queue.pop(0)
            flip = False
            
            if isinstance(node, HTLeaf):
                # Assign -1 since the current node is a leaf node
                hoeffding_dict["children_left"][node_idx] = -1
                hoeffding_dict["children_right"][node_idx] = -1
                # Assign -2 since the current node is a leaf node
                hoeffding_dict["features"][node_idx] = -2
                hoeffding_dict["thresholds"][node_idx] = -2
            else:
                feature = node.feature
                
                # Get the split value and the filters for both left and right child
                if isinstance(node, NumericBinaryBranch):
                    split_threshold = node.threshold
                else: # isinstance(node, NominalBinaryBranch)                    
                    split_threshold = node.value
                    # Check if the left and right child need to be flipped
                    if split_threshold == 1.0:
                        flip = True
                    split_threshold = 0.5
            
                # Update the feature and split value of the current node
                hoeffding_dict["features"][node_idx] = features_idx[feature]
                hoeffding_dict["thresholds"][node_idx] = split_threshold
            
                # Add the child nodes to the queue
                if flip:
                    node_count += 1
                    hoeffding_dict["children_left"][node_idx] = node_count
                    queue.append((node.children[1], node_count))
                    node_count += 1
                    hoeffding_dict["children_right"][node_idx] = node_count
                    queue.append((node.children[0], node_count))
                
                else:
                    node_count += 1
                    hoeffding_dict["children_left"][node_idx] = node_count
                    queue.append((node.children[0], node_count))
                    node_count += 1
                    hoeffding_dict["children_right"][node_idx] = node_count
                    queue.append((node.children[1], node_count))
            
            # Retrieve the prediction value
            value = node.stats
            value_sum = sum(value.values())

            # Check if any class label is missing, this can happened when the 
            # new leaf node has not encounter a class label yet, thus no 
            # observed class weights for that class label
            for i in range(0, 2):
                # Check for zero division
                if i not in value or value_sum == 0:
                    """
                    In order to prevent the error shown below, the value[i] must not be zero when the dictionary is ingested into tree SHAP explainer using 'tree_path_dependent' approach. 

                    AssertionError: The background dataset you provided does not cover all the leaves in the model, so TreeExplainer cannot run with the feature_perturbation="tree_path_dependent" option! Try providing a larger background dataset, no background dataset, or using feature_perturbation="interventional".
                    """
                    value[i] = 1

            # Re-evaluate the sum of prediction value
            value_sum = sum(value.values())

            # Scale down by the total number of base learners
            value = np.array([value[0], value[1]]) * scaling
            # Normalize the stats
            value /= value_sum
            hoeffding_dict["values"][node_idx] = value

            # Assign the sum of weights in stats to node_sample_weight
            hoeffding_dict["node_sample_weight"][node_idx] = value_sum
            
        hoeffding_dict["children_default"] = hoeffding_dict["children_left"]
        hoeffding_dict["values"] = np.reshape(hoeffding_dict["values"], (total_nodes, N_OUTPUTS * NUM_CLASSES))
        arf_dict["trees"].append(hoeffding_dict)

    return arf_dict
