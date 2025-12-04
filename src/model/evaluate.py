from sklearn.metrics import classification_report, confusion_matrix
def evaluate_model(model, X_test, y_test):
    """ 
    Evaluate an XGBoost model on test data.

    Args:
       model : Trained Model
       X_test : Test features.
       y_test : Test labels.
    
    """

    preds = model.predict(X_test)
    print("classification report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))