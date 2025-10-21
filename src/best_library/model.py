from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, model_type='logistic'):

    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(random_state=42)
    
    model.fit(X_train, y_train)
    return model

def predict(model, train_df, test_df, features):
    """Add prediction probabilities to both train and test sets."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df['predictions'] = model.predict_proba(train_df[features])[:, 1]
    test_df['predictions'] = model.predict_proba(test_df[features])[:, 1]

    return train_df, test_df
