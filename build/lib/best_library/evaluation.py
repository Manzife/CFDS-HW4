from sklearn.metrics import roc_auc_score

def compute_roc_auc(y_true, y_pred):
    """Compute ROC AUC score."""
    return roc_auc_score(y_true, y_pred)


