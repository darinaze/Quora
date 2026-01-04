from sklearn.metrics import (
    log_loss,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test, threshold=0.5):
   
    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = []
    for p in y_prob:
        if p >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    results = {
        "threshold": threshold,
        "log_loss": log_loss(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }

    return results



def evaluate_with_threshold(y_true, y_prob, threshold):
    predictions = []
    for p in y_prob:
        if p >= threshold:
            predictions.append(1)
        else:
            predictions.append(0)

    return f1_score(y_true, predictions)
