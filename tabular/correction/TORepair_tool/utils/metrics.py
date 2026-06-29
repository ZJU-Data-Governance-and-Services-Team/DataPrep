import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def classi_metrics(all_preds, all_labels):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(classification_report(all_labels, all_preds))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }