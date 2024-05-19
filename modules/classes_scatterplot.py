import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt


def draw_classes_scatterplot(X, y, title, class_list, colors):
    """
    This function draws scatterplot of classified objects
    :param X: Values of time series
    :param y: class of time series
    :param title: title of graph
    :param class_list: list of classifications classes
    :param colors: list of colors of points in scatterplot
    """

    Z = np.c_[X, y]
    predictions = [ [] for i in range(len(class_list)) ]

    for i in range(Z.shape[0]):
        for _class in range(len(class_list)):
            if Z[:, -1][i] == _class:
                predictions[_class].append(Z[i])
                
    for i in range(len(predictions)):
        predictions[i] = np.array(predictions[i])
        predictions[i] = np.delete(predictions[i], -1, axis=1)
        
    scaler = StandardScaler()
    pca = decomposition.PCA(n_components=3)

    for i in range(len(predictions)):
        scaler.fit(predictions[i])
        predictions[i] = scaler.transform(predictions[i])

        pca.fit(predictions[i])
        predictions[i] = pca.transform(predictions[i])

    plt.figure(figsize=(6, 6))
    ax = plt.axes(projection='3d')

    for i in range(len(predictions)):
        ax.scatter(predictions[i][:, 0], predictions[i][:, 1], predictions[i][:, 2], alpha=0.8, c=colors[i], label=class_list[i])

    plt.title(title)
    plt.legend(loc=2)

    plt.show()