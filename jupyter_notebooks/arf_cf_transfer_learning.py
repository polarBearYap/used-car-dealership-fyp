# Standard libraries
import numpy as np
import pandas as pd

import copy
from joblib import Parallel, delayed
import time

# River components
from river.tree.nodes.branch import NumericBinaryBranch, NominalBinaryBranch

# River utilities
from river.proba import Gaussian

# Author: Yap Jheng Khin

def fetch_node(root_node, location):
    """Fetch a particular node.
    
    This happens when a leaf node has split and turned into parent node.
    The object type of leaf node is different from parent node.
    Thus, the node must be fetched from the root node all over again. 

    Parameters
    ----------
    root_node: river.tree.nodes.HTLeaf or river.tree.nodes.DTBranch
        Root node.
    location: str
        An array consists of 0 and 1. If the value is 0, traverse to left and vice versa.
        
    Returns
    -------
    node : river.tree.nodes.HTLeaf or river.tree.nodes.DTBranch
        Node that is fetched based on the location.
    """
    
    node = root_node
    for path in location:
        node = node.children[int(path)]
    return node

def fetch_node_and_data(root_node, location, df):
    """Filter the training data that corresponds to the node at the given location. 

    Parameters
    ----------
    root_node: river.tree.nodes.HTLeaf or river.tree.nodes.DTBranch
        Root node.
    location: str
        An array consists of 0 and 1. If the value is 0, traverse to left and vice versa.
    df: DataFrame
        Training data.
        
    Returns
    -------
    node : river.tree.nodes.HTLeaf or river.tree.nodes.DTBranch
        Node that is fetched based on the location.
    df: DataFrame
        Data that correspond to the fetched node.
    """
    
    node = root_node
    
    for path in location:
        path = int(path)
        feature = node.feature
        # Filter training data based on current split criteria
        if isinstance(node, NumericBinaryBranch):
            # Split criteria of numeric branch
            index = df[feature] <= node.threshold
            if path == 0:
                df = df[index]
            else:
                df = df[~index]
        elif isinstance(node, NominalBinaryBranch):
            # Split criteria of nominal branch
            index = df[feature] == node.value
            if path == 0:
                df = df[index]
            else:
                df = df[~index]
        node = node.children[path]
    
    return node, df

def filter_data(df, split_feature, split_value, nominal_attrs):
    """Filter the training data that corresponds to the node given the split feature and value. 

    Parameters
    ----------
    df: DataFrame
        Training data that corresponds to a split node.
    split_feature: str
        The split feature of the node.
    split_value: float
        The split value of the node.
    nominal_attrs: list of str
        The list containing the nominal feature names.
        
    Returns
    -------
    df_left : DataFrame
        Data that correspond to the left child of the node.
    df_right: DataFrame
        Data that correspond to the right child of the node.
    """

    df_left = df_right = None

    if split_feature not in nominal_attrs:
        # Split criteria of numeric branch
        index = df[split_feature] <= split_value
        df_left = df[index]
        df_right = df[~index]

    elif split_feature in nominal_attrs:
        # Split criteria of nominal branch
        index = df[split_feature] == split_value
        df_left = df[index]
        df_right = df[~index]

    return [df_left, df_right]

def dt_fetch_data(decision_tree, location, df):
    """Filter the data given the node's location. 
    This fetching algorithm is only for traditional random forest. 
    `fetch_node`, `fetch_node_and_data`, `filter_data` are the fetching algorithms for adaptive random forest.

    Parameters
    ----------
    decision_tree: sklearn.tree.DecisionTreeClassifier
        Decision tree.
    location: str
        An array consists of 0 and 1. If the value is 0, traverse to left and vice versa.
    df: DataFrame
        Data.

    Returns
    -------
    df: DataFrame
        Filtered Data that correspond to the given node's location.
    """

    node_id = 0

    for path in location:
        path = int(path)

        feature = decision_tree.feature[node_id]
        threshold = decision_tree.threshold[node_id]

        # Filter training data based on current split criteria
        index = df.iloc[:, feature] <= threshold
        if path == 0:
            df = df[index]
            node_id = decision_tree.children_left[node_id]
        else:
            df = df[~index]
            node_id = decision_tree.children_right[node_id]
            
    return df

def learn_many(train_data, node, dt_leaf_stats, nominal_attributes):
    """Update the statistics, splitters, prediction weights in a leaf node given the training data.
    
    Parameters
    ----------
    train_data: DataFrame
        The train data that correspond to the current leaf node.
    node: river.tree.nodes.HTLeaf
        The leaf node.
    dt_leaf_stats: dict
        Dictionary containing the corresponding decision tree's leaf node's statistics.
        The dictionary structure is { 0: value, 1: value }, where 0 represent negative class and vice versa.
    nominal_attrs: list of str
        The list containing the nominal feature names.

    Returns
    -------
         
    Credits
    -------
    River. This is the vectorised version of River API implementation.
    """

    # ------------------------------------------- #
    # Update statistics
    # ------------------------------------------- #

    # Set sample weight to 1
    # Note that the Adaptive Random Forest Algorithm randomly generate a integer when training each sample.
    # Thus, a fixed value of 1 is used for consistency.
    # sample_weight = arf_cf._rng.poisson(
    #     lam=arf_cf.lambda_value
    # )
    sample_weight = 1

    POS = 1
    NEG = 0
    if NEG in node.stats:
        node.stats[NEG] += sample_weight * dt_leaf_stats[NEG]
    else:
        node.stats[NEG] = sample_weight * dt_leaf_stats[NEG]
    if POS in node.stats:
        node.stats[POS] += sample_weight * dt_leaf_stats[POS]
    else:
        node.stats[POS] = sample_weight * dt_leaf_stats[POS]
        
    # ------------------------------------------- #
    # Update splitters
    # ------------------------------------------- #
    X = train_data.drop(columns='y', axis=1)
    y = train_data['y'].copy()
    num_samples = len(y)


    for att_id, att_vals in X.iteritems():

        if att_id in node._disabled_attrs:
            continue

        # Get the splitter that corresponds to att_id 
        if att_id in node.splitters:
            splitter = node.splitters[att_id]
        # Initialise any new splitter
        else:
            # This condition assumes that the nominal_attributes are explicitly specified
            if att_id in nominal_attributes:
                splitter = node.new_nominal_splitter()
            else:
                splitter = copy.deepcopy(node.splitter)
            node.splitters[att_id] = splitter

        missing_index = np.isnan(att_vals)
        non_null_target_vals = y[~missing_index]
        non_null_att_vals = att_vals[~missing_index]
        df = pd.DataFrame(np.dstack((non_null_target_vals, non_null_att_vals))[0], 
                          columns=['tar', 'att'])

        # Update nominal binary splitter
        if att_id in nominal_attributes:
            missing_count = np.count_nonzero(missing_index)
            splitter._missing_weight_observed += sample_weight * missing_count
            splitter._att_values.update(np.unique(non_null_att_vals))
            df = df.groupby(['tar','att']).size()
            for tar, df2 in df.groupby(level=0):
                for att, _ in df2.groupby(level=1):
                    final_weight = sample_weight * df[tar][att]
                    splitter._att_dist_per_class[tar][att] += final_weight
            splitter._total_weight_observed += sample_weight * num_samples

        # Update Gaussian binary splitter
        else:
            val_dist = None
            df2 = df.groupby('tar').agg(['min', 'max'])

            for target_class, row in df2.iterrows():
                new_min = df2.loc[target_class]['att']['min']
                new_max = df2.loc[target_class]['att']['max']

                if target_class in splitter._att_dist_per_class:
                    val_dist = splitter._att_dist_per_class[target_class]
                    old_min = splitter._min_per_class[target_class]
                    old_max = splitter._max_per_class[target_class]
                    splitter._min_per_class[target_class] = min(old_min, new_min)
                    splitter._max_per_class[target_class] = max(old_max, new_max)
                else:
                    val_dist = Gaussian()
                    splitter._att_dist_per_class[target_class] = val_dist
                    splitter._min_per_class[target_class] = new_min
                    splitter._max_per_class[target_class] = new_max

                val_dist._var.update_many(non_null_att_vals)

def get_stats(stats, label_classes):
    """
    Convert the prediction value retrieved from the decision tree's node
    into the dictionary. The dictionary will be passed as a parameter
    into `learn_many` or assigned to branch nodes' stats attribute.

    Parameters
    ----------
    stats: list
        List containing the decision tree's node's prediction value.
    label_classes: list-like object
        List containing the label classes that map the prediction values 
        to the respective label classes.

    Returns
    -------
    new_stats: dict
        Dictionary containing label classes mapped to prediction values.
    """
    new_stats = {}

    for idx in range(len(stats)):
        new_stats.update({label_classes[idx]: stats[idx]})

    return  new_stats

def transfer_tree_weights(task_no, hoeffding_tree, decision_tree, train_data, label_classes, 
                          feature_names, nominal_attrs):
    """Transfer the trained weights from the decision tree to the hoeffding tree.
    
    Parameters
    ----------
    task_no: int
        Task number to trace the original order of trees in the ensemble.
    hoeffding_tree: river.tree.HoeffdingTree
        The object representing the Hoeffding tree.
    decision_tree: sklearn.tree.DecisionTreeClassifier
        The object representing the decision tree.
    train_data: DataFrame
        Training data.
    label_classes: list of int or list of str
        The list containing the class label names.
    feature_names: list of str
        The list containing the X feature names.
    nominal_attrs: list of str
        The list containing the nominal feature names.

    Returns
    -------
    task_no: int
        Task number to trace the original order of trees in the ensemble.
    hoeffding_tree: river.tree.HoeffdingTree
        The Hoeffding tree's object with the transferred weights.
    """
    # Update the set of observed classes
    hoeffding_tree.classes = set(np.unique(label_classes))
    decision_tree = decision_tree.tree_
    # Initialise the root node
    hoeffding_tree._root = hoeffding_tree._new_leaf()
    hoeffding_tree._n_active_leaves = 1

    # Each queue elements: (current node index, current location, current depth)
    # New element is appended when a new node can be further splitted
    queue = [(0, '', 0)]

    while len(queue) > 0:
        node_idx, node_location, cur_depth = queue.pop(0)
        left_child_idx = decision_tree.children_left[node_idx]
        right_child_idx = decision_tree.children_right[node_idx]

        # Get the parent of the currrent node
        parent_node, df_parent = fetch_node_and_data(hoeffding_tree._root, node_location[:-1], train_data)
        # Get the currrent node
        if len(node_location) == 0:
            tar_node_location = ''
        else:
            tar_node_location =  node_location[-1]
        cur_node, df_cur = fetch_node_and_data(parent_node, tar_node_location, df_parent)

        cur_node_stats = decision_tree.value[node_idx][0]
        cur_node_stats = get_stats(cur_node_stats, label_classes)

        # Split the current node if the corresponding decision tree's node is a parent node   
        if left_child_idx != -1:
            split_feature_idx = decision_tree.feature[node_idx]
            split_feature = feature_names[split_feature_idx]
            split_threshold = decision_tree.threshold[node_idx]
            # Change split value from 0.5 to 0 for nominal attributes
            # since decision tree and hoeffding tree work slightly differently
            if split_feature in nominal_attrs:
                split_threshold = 0.0

            # Split the current node
            child_nodes = [
                hoeffding_tree._new_leaf({}, cur_node)
                for _ in range(0, 2)
            ]

            # Determine the branch type based on the type of split feature
            if split_feature in nominal_attrs:
                branch_type = NominalBinaryBranch
            else:
                branch_type = NumericBinaryBranch

            # Create a new branch
            new_split_node = branch_type({}, split_feature, split_threshold, cur_depth, *child_nodes)
            hoeffding_tree._n_active_leaves += 1
            new_split_node.stats = cur_node_stats

            # Attach the split node directly to the root reference if there is no parent
            if node_location == '':
                hoeffding_tree._root = new_split_node
            # Else, connect the split node with its parent
            else:
                parent_node.children[int(node_location[-1])] = new_split_node

            # Append the leaf nodes to the queue
            for p_branch, child_idx in enumerate([left_child_idx, right_child_idx]):
                new_node_location = node_location + str(p_branch)
                queue.append((child_idx, new_node_location, cur_depth+1))

        else:
            # Update the statistics information
            learn_many(df_cur, cur_node, cur_node_stats, nominal_attrs)

    return (task_no, hoeffding_tree)

def transfer_tree_weights_test(hoeffding_tree, decision_tree, train_data, label_classes, 
                               feature_names, nominal_attrs, log_file_path='outputs/cf_transfer_learning/arf_debug.txt'):
    """Transfer the trained weights from the decision tree to the hoeffding tree. 
    This is the test version of the `transfer_tree_weights` to validate the algorithm.
    
    Parameters
    ----------
    hoeffding_tree: river.tree.HoeffdingTree
        The object representing the Hoeffding tree.
    decision_tree: sklearn.tree.DecisionTreeClassifier
        The object representing the decision tree.
    train_data: DataFrame
        Training data.
    label_classes: list of int or list of str
        The list containing the class label names.
    feature_names: list of str
        The list containing the X feature names.
    nominal_attrs: list of str
        The list containing the nominal feature names.
    log_file_path: str optional
        The file path to write the log delimited by forward slash character.

    Returns
    -------

    """

    # Update the set of observed classes
    hoeffding_tree.classes = set(np.unique(label_classes))
    decision_tree = decision_tree.tree_
    # Initialise the root node
    hoeffding_tree._root = hoeffding_tree._new_leaf()
    hoeffding_tree._n_active_leaves = 1

    # Each queue elements: (current node index, current location, current depth)
    # New element is appended when a new node can be further splitted
    queue = [(0, '', 0)]

    duration = 0
    start = time.time()

    with open(log_file_path, 'w') as f:
        while len(queue) > 0:
            node_idx, node_location, cur_depth = queue.pop(0)
            left_child_idx = decision_tree.children_left[node_idx]
            right_child_idx = decision_tree.children_right[node_idx]

            # Get the parent of the currrent node
            parent_node, df_parent = fetch_node_and_data(hoeffding_tree._root, node_location[:-1], train_data)
            # Get the currrent node
            if len(node_location) == 0:
                tar_node_location = ''
            else:
                tar_node_location =  node_location[-1]
            cur_node, df_cur = fetch_node_and_data(parent_node, tar_node_location, df_parent)

            # Split the current node if the corresponding decision tree's node is a parent node   
            if left_child_idx != -1:
                split_feature_idx = decision_tree.feature[node_idx]
                split_feature = feature_names[split_feature_idx]
                split_threshold = decision_tree.threshold[node_idx]
                # Change split value from 0.5 to 0 for nominal attributes
                # since decision tree and hoeffding tree work slightly differently
                if split_feature in nominal_attrs:
                    split_threshold = 0.0

                # Split the current node
                child_nodes = [
                    hoeffding_tree._new_leaf({}, cur_node)
                    for _ in range(0, 2)
                ]

                # Determine the branch type based on the type of split feature
                if split_feature in nominal_attrs:
                    branch_type = NominalBinaryBranch
                else:
                    branch_type = NumericBinaryBranch

                # Create a new branch
                new_split_node = branch_type({}, split_feature, split_threshold, cur_depth, *child_nodes)
                hoeffding_tree._n_active_leaves += 1
                parent_stats = decision_tree.value[node_idx][0]
                parent_stats = get_stats(parent_stats, label_classes)
                new_split_node.stats = parent_stats

                # Attach the split node directly to the root reference if there is no parent
                if node_location == '':
                    hoeffding_tree._root = new_split_node
                # Else, connect the split node with its parent
                else:
                    parent_node.children[int(node_location[-1])] = new_split_node

                # Fetch the training data that corresponds with the current node,
                # assuming if the current node has splitted
                df_children = filter_data(df_cur, split_feature, split_threshold, nominal_attrs)
                f.write(f'Node location: {node_location}\n')
                f.write(f'Left  child index: {left_child_idx}\nCount: {df_children[0].shape[0]}\n')
                f.write(f'Right child index: {right_child_idx}\nCount: {df_children[1].shape[0]}\n\n')

                # Append the leaf nodes to the queue
                for p_branch, child_idx in enumerate([left_child_idx, right_child_idx]):
                    new_node_location = node_location + str(p_branch)
                    queue.append((child_idx, new_node_location, cur_depth+1))

            else:
                # Update the statistics information
                leaf_stats = decision_tree.value[node_idx][0]
                leaf_stats = get_stats(leaf_stats, label_classes)
                learn_many(df_cur, cur_node, leaf_stats, nominal_attrs)

    end = time.time()
    duration += end - start
    print(f'\nTime taken for testing the transfer learning: {duration} seconds')

def transfer_learning(arf_cf, random_forest, train_data, feature_names, nominal_attrs):
    """Transfer the trained weights from the traditional random forest to the adaptive random forest.
    
    Parameters
    ----------
    arf_cf: river.ensemble.BaseForest
        The object representing the adaptive random forest.
    random_forest: sklearn.ensemble.RandomForestClassifier
        The object representing the traditional random forest.
    train_data: DataFrame
        Training data.
    feature_names: list of str
        The list containing the feature names.
    nominal_attrs: list of str
        The list containing the nominal feature names.

    Returns
    -------
    arf_cf: river.ensemble.BaseForest
        The adaptive random forest's object with the transfered weights.
    """
    
    # Get classes
    classes = random_forest.classes_
    
    duration = 0
    start = time.time()
    
    # Break down into smaller task, where each task transfer weights between a pair of trees
    # Spawn multiple processes to run tasks in parallel
    trees_transfered = Parallel(
        n_jobs=-1,
        verbose=11,
        prefer="processes",
    )(
        delayed(transfer_tree_weights)(
            task_no,
            hoeffding_tree_member.model,
            decision_tree,
            train_data,
            classes, 
            feature_names, 
            nominal_attrs
        )
        for task_no, (decision_tree, hoeffding_tree_member) in enumerate(zip(random_forest.estimators_, arf_cf))
    )
    
    end = time.time()
    duration += end - start
    print(f'\nTime taken for performing transfer learning: {duration} seconds')
    
    # Update the tree objects in the ensemble respectively by refering to the task_no
    for index, tree in trees_transfered:
        arf_cf.data[index].model = tree
        
    return arf_cf