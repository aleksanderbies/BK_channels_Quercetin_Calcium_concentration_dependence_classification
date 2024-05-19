from sklearn.model_selection import train_test_split, StratifiedKFold
import scikitplot as skplt
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from scikitplot.estimators import plot_learning_curve
import matplotlib.pyplot as plt
import numpy as np

def run_model(model, X, y, scoring="accuracy", cv=StratifiedKFold(n_splits=4, random_state=0, shuffle=True)):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f" CV Scores: {scores}")
    print(f" CV Mean score: {np.mean(scores)}")

    skplt.estimators.plot_learning_curve(model, X, y, cv=cv, scoring=scoring, figsize=(15, 5));

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    y_pred = cross_val_predict(model, X, y)
    skplt.metrics.plot_confusion_matrix(y, y_pred, normalize=False, ax=ax1)

    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    try:
        y_pred_proba = model.predict_proba(X)
        skplt.metrics.plot_roc(y, y_pred_proba, ax=ax2)
    except AttributeError:
        print("This classifier has no attribute 'predict_proba' ")
        
def find_n_neighbors(model, X, y, n):
    k_values = range(1, n)
    accuracy_values = []

    for k in k_values:
        model.set_params(n_neighbors=k)        
        model.fit(X, y)
        accuracy = model.score(X, y)
        accuracy_values.append(accuracy)

    plt.figure(figsize=(8,4))    
    plt.plot(k_values, accuracy_values, marker='o')
    plt.title('Classification accuracy depending on the number of neighbors (k)')
    plt.xlabel('Number of neighbors (k)')
    plt.ylabel('Accuracy score')
    plt.show()
    
def test_model(model, X, y, scoring="accuracy", cv=StratifiedKFold(n_splits=4, random_state=0, shuffle=True)):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f" CV Scores: {scores}")
    print(f" CV Mean score: {np.mean(scores)}")
    
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    y_pred = cross_val_predict(model, X, y)
    skplt.metrics.plot_confusion_matrix(y, y_pred, normalize=False, ax=ax1)

    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    try:
        y_pred_proba = model.predict_proba(X)
        skplt.metrics.plot_roc(y, y_pred_proba, ax=ax2)
    except AttributeError:
        print("This classifier has no attribute 'predict_proba' ")