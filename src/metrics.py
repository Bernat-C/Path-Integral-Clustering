from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def clustering_error(y_true, y_pred):
        """
        Computes the clustering error
        
        Parameters:
            y_true (numpy.ndarray): Ground truth labels.
            y_pred (numpy.ndarray): Cluster labels from the algorithm.
        
        Returns:
            float: Clustering error (misclassification rate).
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Solve the assignment problem (Hungarian algorithm) to maximize correct assignments
        row_ind, col_ind = linear_sum_assignment(-cm)  # Maximize correct pairs

        # Compute the total number of correctly assigned points
        correct = cm[row_ind, col_ind].sum()
        
        # Compute clustering error
        error = 1 - (correct / len(y_true))
        return error