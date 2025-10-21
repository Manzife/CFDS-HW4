from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, model_type='logistic'):

    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(random_state=42)
    
    model.fit(X_train, y_train)
    return model

def predict(model, X_train, X_test):
    X_train['predictions'] = model.predict_proba(X_train)[:, 1]
    X_test['predictions'] = model.predict_proba(X_test)[:, 1]
    return X_train, X_test