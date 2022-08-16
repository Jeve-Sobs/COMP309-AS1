# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random

import sklearn
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN, Birch, AgglomerativeClustering, MeanShift
from sklearn.datasets import make_blobs, make_classification, make_circles
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import openml
import pandas as pd
import numpy as np

def errorRate(Y_pred,Y_test):
    if len(Y_pred) != len(Y_test):
        print("Error list are of different sizes")
    errorCount = 0
    for x in range(len(Y_pred)):
        if Y_pred[x] != Y_test[x]:
            errorCount+=1

    return errorCount

def KNN(X_train, X_test, Y_train, Y_test,n_neighbors):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors)  # , weights="distance")# by default-"uniform"
    neigh.fit(X_train, Y_train)
    Y_pre = neigh.predict(X_test)
    return (1- np.mean(Y_pre != Y_test))*100

def DT(X_train, X_test, Y_train, Y_test, max_depth):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(X_train, Y_train)
    Y_pre = clf.predict(X_test)
    return (1- np.mean(Y_pre != Y_test))*100

def LogRegression(X_train, X_test, Y_train, Y_test,c):
    logreg = LogisticRegression(C=c)
    logreg.fit(X_train, Y_train)
    Y_pre = logreg.predict(X_test)
    return (1- np.mean(Y_pre != Y_test))*100

def NB(X_train, X_test, Y_train, Y_test,var):
    GNBclf = GaussianNB(var_smoothing = var)
    GNBclf.fit(X_train, Y_train)
    Y_pre = GNBclf.predict(X_test)
    return (1- np.mean(Y_pre != Y_test))*100

def GB(X_train, X_test, Y_train, Y_test,depth):
    reg = GradientBoostingClassifier(max_depth = depth)
    reg.fit(X_train, Y_train)
    Y_pre = reg.predict(X_test)
    return (1- np.mean(Y_pre != Y_test))*100

def RF(X_train, X_test, Y_train, Y_test,depth):
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, Y_train)
    Y_pre = clf.predict(X_test)
    return (1- np.mean(Y_pre != Y_test))*100


def MLP(X_train, X_test, Y_train, Y_test,alpha):
    clf = MLPClassifier(alpha = alpha, max_iter=2000)
    clf.fit(X_train, Y_train)
    Y_pre = clf.predict(X_test)
    return (1- np.mean(Y_pre != Y_test))*100

def display(data,title,parmas,dataset,hyperParameter):
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    axes = fig.add_subplot(111)
    bp = axes.boxplot(data)
    if len(parmas) == 3:
        plt.xticks([1,2,3], parmas)
    if len(parmas) == 4:
        plt.xticks([1,2,3,4], parmas)
    if len(parmas) == 5:
        plt.xticks([1,2,3,4,5], parmas)

    axes.set_xlabel('HyperParameter '+hyperParameter)
    axes.set_ylabel('Accuracy percentage')
    axes.set_title(title+" classification using "+ dataset + " dataset")
    # show plot
    #plt.show()
    fig.savefig('/Users/felixnagel/Documents/ml-as1-photos/smart-way/'+title+"-"+ dataset + ".png")  # save the figure to file
    plt.close(fig)  # close the figure window

def KMeans_Ckust(X):
    clust = KMeans()
    return clust.fit_predict(X)

def AP_Clust(X):
    AP = AffinityPropagation()
    y = AP.fit_predict(X)
    return y

def DBSCAN_Clust(X):
    clust = DBSCAN()
    y = clust.fit_predict(X)
    return y

def GM_Clust(X):
    clust = GaussianMixture()
    y= clust.fit_predict(X)
    return y

def birch_Clust(X):
    clust = Birch()
    return clust.fit_predict(X)

def A_Clust(X):
    clust = AgglomerativeClustering()
    return clust.fit_predict(X)

def MS_Clust(X):
    clust = MeanShift()
    return clust.fit_predict(X)

#Save and display scatter plot
def scatter(x,y,model,dataset):
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.title(model +" Clustering using "+dataset)
    plt.savefig('/Users/felixnagel/Documents/ml-as1-photos/clustering/'+model+'-'+dataset)
    plt.show()

def clustering():
    K_Means_bool = True
    AffinityPropagation_bool = False
    DBSCAN_bool = False
    GaussianMixture_bool = False
    BIRCH_bool = False
    AgglomerativeClustering_bool = False
    MeanShift_bool = False

    if True:
        X, y = make_blobs(n_samples=1000,random_state=101,n_features=2)
        dataset = "Make Blobs Dataset"

    if False:
        X, y = make_classification(n_clusters_per_class=1, n_samples=1000,random_state=101,n_redundant = 0, n_features=2)
        dataset = "Make Classification Dataset"

    if False:
        X, y = make_circles(noise=0.3, n_samples=1000,random_state=101)
        dataset = "Make Circles Dataset"

    X = np.array(X)

    if K_Means_bool:
        model = "K_Means"
        KMeans_Ckust(X)
    if AffinityPropagation_bool:
        model = "Affinity Propagation"
        AP_Clust(X)
    if DBSCAN_bool:
        model = 'DBSCAN'
        DBSCAN_Clust(X)
    if GaussianMixture_bool:
        model = "Gaussian Mixture"
        y = GM_Clust(X)
    if BIRCH_bool:
        model = "BIRCH"
        y = birch_Clust(X)
    if AgglomerativeClustering_bool:
        model = "Agglomerative"
        y = A_Clust(X)
    if MeanShift_bool:
        model = "MeanShift"
        y = MS_Clust(X)

    scatter(X, y, model , dataset)

def main(x, labels, datasetName):

    SavePlots = False
    PrintMean = True

    KNN_bool = False
    NB_bool = False
    LogReg_bool = False
    DT_bool = False
    GB_bool = False
    RF_bool = False
    MLP_bool = True

    Temp_List1 = []
    Temp_List2 = []
    Temp_List3 = []
    Temp_List4 = []
    Temp_List5 = []

    for i in range(50):
        r = random.randrange(1000)
        X_train, X_test, Y_train, Y_test = train_test_split(x, labels, test_size=0.5, random_state=r)

        # KNN
        if KNN_bool:
            Temp_List1.append(KNN(X_train, X_test, Y_train, Y_test, 1))
            Temp_List2.append(KNN(X_train, X_test, Y_train, Y_test, 2))
            Temp_List3.append(KNN(X_train, X_test, Y_train, Y_test, 3))
            Temp_List4.append(KNN(X_train, X_test, Y_train, Y_test, 4))
            Temp_List5.append(KNN(X_train, X_test, Y_train, Y_test, 5))

        # GaussianNB
        if NB_bool:
            Temp_List1.append(NB(X_train, X_test, Y_train, Y_test, 0.1))
            Temp_List2.append(NB(X_train, X_test, Y_train, Y_test, 0.00001))
            Temp_List3.append(NB(X_train, X_test, Y_train, Y_test, 0.000000001))

        # LogisticRegression
        if LogReg_bool:
            Temp_List1.append(LogRegression(X_train, X_test, Y_train, Y_test, 0.1))
            Temp_List2.append(LogRegression(X_train, X_test, Y_train, Y_test, 0.5))
            Temp_List3.append(LogRegression(X_train, X_test, Y_train, Y_test, 1))
            Temp_List4.append(LogRegression(X_train, X_test, Y_train, Y_test, 2))
            Temp_List5.append(LogRegression(X_train, X_test, Y_train, Y_test, 5))

        # DecisionTreeClassifier
        if DT_bool:
            Temp_List1.append(DT(X_train, X_test, Y_train, Y_test, 1))
            Temp_List2.append(DT(X_train, X_test, Y_train, Y_test, 3))
            Temp_List3.append(DT(X_train, X_test, Y_train, Y_test, 5))
            Temp_List4.append(DT(X_train, X_test, Y_train, Y_test, 8))
            Temp_List5.append(DT(X_train, X_test, Y_train, Y_test, 10))

        # GradientBoostingClassifier
        if GB_bool:
            Temp_List1.append(GB(X_train, X_test, Y_train, Y_test, 1))
            Temp_List2.append(GB(X_train, X_test, Y_train, Y_test, 3))
            Temp_List3.append(GB(X_train, X_test, Y_train, Y_test, 5))
            Temp_List4.append(GB(X_train, X_test, Y_train, Y_test, 8))
            Temp_List5.append(GB(X_train, X_test, Y_train, Y_test, 10))

        # RandomForestClassifier
        if RF_bool:
            Temp_List1.append(RF(X_train, X_test, Y_train, Y_test, 1))
            Temp_List2.append(RF(X_train, X_test, Y_train, Y_test, 3))
            Temp_List3.append(RF(X_train, X_test, Y_train, Y_test, 5))
            Temp_List4.append(RF(X_train, X_test, Y_train, Y_test, 8))
            Temp_List5.append(RF(X_train, X_test, Y_train, Y_test, 10))

        # MLPClassifier
        if MLP_bool:
            Temp_List1.append(MLP(X_train, X_test, Y_train, Y_test, 0.00001))
            Temp_List2.append(MLP(X_train, X_test, Y_train, Y_test, 0.001))
            Temp_List3.append(MLP(X_train, X_test, Y_train, Y_test, 0.1))
            Temp_List4.append(MLP(X_train, X_test, Y_train, Y_test, 10))



    if PrintMean:

        maxAr = []
        maxAr.append(np.mean(Temp_List1))
        maxAr.append(np.mean(Temp_List2))
        maxAr.append(np.mean(Temp_List3))
        maxAr.append(np.mean(Temp_List4))
        #maxAr.append(np.mean(Temp_List5))
        maxMean = np.max(maxAr)
        if np.mean(Temp_List1) == maxMean:
            print("Number 1 hyper parameter")
        if np.mean(Temp_List2) == maxMean:
            print("Number 2 hyper parameter")
        if np.mean(Temp_List3) == maxMean:
            print("Number 3 hyper parameter")
        if np.mean(Temp_List4) == maxMean:
            print("Number 4 hyper parameter")
        if np.mean(Temp_List5) == maxMean:
            print("Number 5 hyper parameter")

        print(round(maxMean,3))

    # Display graphs
    if SavePlots:
        if KNN_bool:
            display([Temp_List1, Temp_List2, Temp_List3, Temp_List4, Temp_List5, ], "KNeighborsClassifier",
                    [1, 2, 3, 4, 5], datasetName, "n neighbors")
        if NB_bool:
            display([Temp_List1, Temp_List2, Temp_List3], "GaussianNB", [0.1, 0.00001, 0.000000001], datasetName,
                    "var smoothing")
        if LogReg_bool:
            display([Temp_List1, Temp_List2, Temp_List3, Temp_List4, Temp_List5], "LogisticRegression",
                    [0.1, 0.5, 1.0, 2.0, 5.0], datasetName, "C")
        if DT_bool:
            display([Temp_List1, Temp_List2, Temp_List3, Temp_List4, Temp_List5], "DecisionTreeClassifier",
                    [1, 3, 5, 8, 10], datasetName, "max depth")
        if GB_bool:
            display([Temp_List1, Temp_List2, Temp_List3, Temp_List4, Temp_List5], "GradientBoostingClassifier",
                    [1, 3, 5, 8, 10], datasetName, "max depth")
        if RF_bool:
            display([Temp_List1, Temp_List2, Temp_List3, Temp_List4, Temp_List5], "RandomForestClassifier",
                    [1, 3, 5, 8, 10], datasetName, "max depth")
        if MLP_bool:
            display([Temp_List1, Temp_List2, Temp_List3, Temp_List4], "MLPClassifier", [0.00001, 0.001, 0.1, 10],
                    datasetName, "alpha")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    clustering_bool = True
    if clustering_bool:
        clustering()
    # banknote-authentication
    if False:
        dataset = openml.datasets.get_dataset(1462)

        # Get the data itself as a dataframe (or otherwise)
        data = dataset.get_data(dataset_format="dataframe")
        data = pd.DataFrame(data[0])
        x = data.iloc[:, [0, 1, 2, 3]].to_numpy()
        labels = data.iloc[:, [4]].to_numpy().ravel()
        main(x, labels,"banknote-authentication")

    #  steel-plates-fault
    if False:
        dataset = openml.datasets.get_dataset(1504)

        # Get the data itself as a dataframe (or otherwise)
        data = dataset.get_data(dataset_format="dataframe")
        data = pd.DataFrame(data[0])
        x = data.iloc[:, [0, 1, 2, 3, 5, 6]].to_numpy()
        labels = data.iloc[:, [33]].to_numpy().ravel()
        main(x, labels,"steel-plates-fault")

    # ionosphere
    if False:
        dataset = openml.datasets.get_dataset(59)

        # Get the data itself as a dataframe (or otherwise)
        data = dataset.get_data(dataset_format="dataframe")
        data = pd.DataFrame(data[0])
        x = data.iloc[:, [0, 2, 3, 4, 5]].to_numpy()
        labels = data.iloc[:, [34]].to_numpy().ravel()
        main(x, labels,"ionosphere")
