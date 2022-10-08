import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    p = y.sum(axis=0) / y.sum()
    
    return -(p * np.log(p + EPS)).sum()
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    p = y.sum(axis=0) / y.sum()
    
    return 1 - (p**2).sum()
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    mean = y.mean()
    size = len(y)
    
    return ((y - mean)**2).sum() / size

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    mean = y.mean()
    size = len(y)
    
    return np.abs(y - mean).sum() / size


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j >= threshold
        """

        length = X_subset.shape[0]

        X_left = np.zeros_like(X_subset)
        X_right = np.zeros_like(X_subset)
        y_left = np.zeros_like(y_subset)
        y_right = np.zeros_like(y_subset)
        
        left_size = 0
        right_size = 0
        for i in range(length):
            if X_subset[i][feature_index] < threshold:
                X_left[left_size] = X_subset[i]
                y_left[left_size] = y_subset[i]
                left_size += 1
            else:
                X_right[right_size] = X_subset[i]
                y_right[right_size] = y_subset[i]
                right_size += 1

        for i in range(left_size):
            X_right = np.delete(X_right, right_size, 0)
            y_right = np.delete(y_right, right_size, 0)
        for i in range(right_size):
            X_left = np.delete(X_left, left_size, 0)
            y_left = np.delete(y_left, left_size, 0)

        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        length = X_subset.shape[0]

        y_left = np.zeros_like(y_subset)
        y_right = np.zeros_like(y_subset)
        
        left_size = 0
        right_size = 0
        for i in range(length):
            if X_subset[i][feature_index] < threshold:
                y_left[left_size] = y_subset[i]
                left_size += 1
            else:
                y_right[right_size] = y_subset[i]
                right_size += 1

        for i in range(left_size):
            y_right = np.delete(y_right, right_size, 0)
        for i in range(right_size):
            y_left = np.delete(y_left, left_size, 0)
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        
        n_features = X_subset.shape[1]
        n_objects = X_subset.shape[0]
        
        impurity_func, flag = self.all_criterions[self.criterion_name]
        y = y_subset
        if flag == True:
            y = one_hot_encode(n_features, y)
        feature_index = None
        threshold_ret = None
        best_impurity = 10
        for feature_idx in range(n_features):
            if X_subset[:, feature_idx].size == 0:
                continue
            threshold = X_subset[:, feature_idx].mean()
            #values_sorted = np.sort(X_subset[:, feature_idx])
            #thresholds = np.array([(values_sorted[i] + values_sorted[i + 1]) / 2 for i in range(n_objects - 1)])
            #for threshold in thresholds:
            y_left, y_right = self.make_split_only_y(feature_idx, threshold, X_subset, y)
            left_size = y_left.shape[0]
            right_size = y_right.shape[0]
            current_impurity = (left_size * impurity_func(y_left) + right_size * impurity_func(y_right)) / n_objects
            if best_impurity > current_impurity:
                best_impurity = current_impurity
                feature_index = feature_idx
                threshold_ret = threshold
        if feature_index == None:
            feature_index = 0
            while feature_index < n_features and X_subset[:, feature_idx].size == 0:
                feature_index += 1
            if feature_index == n_features:
                return 0, 0
            threshold_ret = X_subset[:, feature_idx].mean()
        return feature_index, threshold_ret
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        root_node = Node(feature_index, threshold)
        if self.depth < self.max_depth:
            (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
            self.depth += 1
            root_node.left_child = self.make_tree(X_left, y_left)
            root_node.right_child = self.make_tree(X_right, y_right)
            self.depth -= 1
        else:
            y = y_subset
            if self.all_criterions[self.criterion_name][1] == False:
                y = one_hot_encode(y_subset)
            root_node.proba = y.sum(axis=0).argmax()
        return root_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        n_objects = X.shape[0]
        y_predicted = np.zeros((n_objects, 1))
        
        for i in range(n_objects):
            current_depth = 0
            current_node = self.root
            while current_depth < self.max_depth and current_node != None:
                if X[int(i), int(current_node.feature_index)] < current_node.value:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child
                current_depth += 1
            y_predicted[i][0] = current_node.proba
#         if self.all_criterions[self.criterion_name][1] == True:
#             return one_hot_encode(X.shape[0], y_predicted)
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        y_predicted = self.predict(X)
        unique, counts = np.unique(y_predicted, return_counts=True)
        overall_count = counts.sum()
        y_probs = count / overall_count
        y_predicted_probs = one_hot_decode(y_probs)
        
        return y_predicted_probs
