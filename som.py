import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from minisom import MiniSom
import numpy as np
import seaborn as sns
from sklearn import preprocessing

def read_data():
    raw_data = pd.read_csv('Datathon2019/Datathon_dataset/mean_train_final.csv')
    X = raw_data[['nlr', 'lactate', 'bcd', 'GENDER', 'age', 'DISCHARGE_LOCATION', 'Neoplastic_disease', 'LIVER_DISEASE', 'CONGESTIVE_HEART_FAILURE', 'OTHER_NEUROLOGICAL', 'RENAL_FAILURE', 'total', 'resp_rate', 'sys_bp', 'temp', 'hr', 'bun', 'sodium', 'glucose_blood', 'hematocrit', 'o2', 'glucose_pleural', 'pH', 'potassium', 'cre', 'wbc', 'platelets', 'alb', 'row_num']]
    y = np.ravel(raw_data[['die_in_h']])
    
    return X, y

def plot3d(datalist, labels):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter([x[8] for x in datalist], [x[10] for x in datalist], [x[13] for x in datalist], c=labels)
    plt.show()

def main():
    x, y = read_data()
    # x = x.values.tolist()
    # kmeans = KMeans(n_clusters=4)
    # kmeans = kmeans.fit(x)
    # labels = kmeans.predict(x)
    # centroids = kmeans.cluster_centers_


    # print('centroids are: \n', centroids)
    # plot3d(x, labels)

    x = preprocessing.StandardScaler().fit(x).transform(x)

    som = MiniSom(20 ,20, 29, sigma=0.5, learning_rate=0.5)
    som.train_random(x, 2000)
    result1 = som.activation_response(x)
    print('result1 are: \n', result1)

    f, (ax1,ax2) = plt.subplots(figsize = (10, 8),nrows=2)
    cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
    sns.heatmap(result1, linewidths = 0.05, ax = ax1, vmax=15000, vmin=0, cmap=cmap)
    plt.show()


main()