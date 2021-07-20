import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from kneed import KneeLocator

class KmeansModel:
    def __init__(self, X, maxRange):
        self.X = X
        self.K = 0
        self.distortions = []
        self.inertias = []
        self.dist_map = {}
        self.iner_map = {}
        self.maxRange = maxRange

        for k in range(1, self.maxRange):
            # Building and fitting the model
            kmeanModel = KMeans(n_clusters=k).fit(self.X)
            kmeanModel.fit(self.X)

            self.distortions.append(sum(np.min(cdist(self.X, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / self.X.shape[0])
            self.inertias.append(kmeanModel.inertia_)

            self.dist_map[k] = sum(np.min(cdist(self.X, kmeanModel.cluster_centers_,
                                           'euclidean'), axis=1)) / self.X.shape[0]
            self.iner_map[k] = kmeanModel.inertia_

        kl = KneeLocator(range(1, self.maxRange), self.inertias, curve = "convex", direction = "decreasing")
        self.K = kl.elbow
        self.calculating_silhouette()

    def plotting_k_elbow_method(self):
        # for key, val in self.dist_map.items():
        #     print(f'{key} : {val}')

        plt.plot(range(1, self.maxRange), self.distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.show()

        # for key, val in self.iner_map.items():
        #     print(f'{key} : {val}')

        plt.plot(range(1, self.maxRange), self.inertias, 'rx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia')
        plt.show()

    def calculating_silhouette(self):
        # A list holds the silhouette coefficients for each k
        silhouette_coefficients = []
        # Notice you start at 2 clusters for silhouette coefficient
        for k in range(2, self.maxRange+1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.X)
            score = silhouette_score(self.X, kmeans.labels_)
            silhouette_coefficients.append(score)
        self.sil_coef = silhouette_coefficients

    def plotting_silhouette(self):
        plt.plot(range(2, 11), self.sil_coef)
        plt.xticks(range(2, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Coefficient")
        plt.show()

def standardize(df_training_):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_training_)
    df_norm = pd.DataFrame(x_scaled)
    # print(df_norm.shape)
    return df_norm

def kmeans_method(df_norm):
    # Trying fit_predict
    kmeans = KMeans(n_clusters=3,random_state=5) # Lu's
    # kmeans = KMeans(n_clusters=3, random_state=134) # Greg's
    label = kmeans.fit_predict(df_norm)

    centroids = kmeans.cluster_centers_
    return label, centroids

def convert_centroids_toDF(centroids):
    print(type(centroids), len(centroids))
    df_centroids = pd.DataFrame(centroids)
    return df_centroids

def plotting_clusters_plotly(df_norm, label, parameters):
    filtered_label0 = df_norm[label == 0]
    filtered_label1 = df_norm[label == 1]
    filtered_label2 = df_norm[label == 2]

    # print(filtered_label0[0],type(filtered_label0))

    scatter = dict(
        mode="markers",
        name="y",
        type="scatter3d",
        x=filtered_label0.iloc[0].append(filtered_label1[0]).append(filtered_label2[0]),
        y=filtered_label0.iloc[1].append(filtered_label1[1]).append(filtered_label2[1]),
        z=filtered_label0.iloc[2].append(filtered_label1[2]).append(filtered_label2[2]),
        marker=dict(size=2, color="rgb(23, 190, 207)")
    )
    clusters = dict(
        alphahull=7,
        name="y",
        opacity=0.1,
        type="mesh3d",
        x=filtered_label0.iloc[0].append(filtered_label1[0]).append(filtered_label2[0]),
        y=filtered_label0.iloc[1].append(filtered_label1[1]).append(filtered_label2[1]),
        z=filtered_label0.iloc[2].append(filtered_label1[2]).append(filtered_label2[2]),
    )
    layout = dict(
        title='3d point clustering',
        scene=dict(
            xaxis=dict(zeroline=False),
            yaxis=dict(zeroline=False),
            zaxis=dict(zeroline=False),
            xaxis_title=parameters[0],
            yaxis_title=parameters[1],
            zaxis_title=parameters[2]),
        # title='KMeans Clustering displayed on a 3-dimensional space',
    )
    fig = dict(data=[scatter, clusters], layout=layout)
    # Use py.iplot() for IPython notebook
    return fig

def plotting_clusters_plotly_with_centroids(df_norm, label, centroids, parameters):
    from plotly.offline import init_notebook_mode, iplot

    filtered_label0 = df_norm[label == 0]
    filtered_label1 = df_norm[label == 1]
    filtered_label2 = df_norm[label == 2]

    scatter1 = dict(
        mode="markers",
        name=parameters[0],
        type="scatter3d",
        x=filtered_label0[0],
        y=filtered_label0[1],
        z=filtered_label0[2],
        marker=dict(size=2,
                    color="red")
        )

    scatter2 = dict(
        mode="markers",
        name=parameters[1],
        type="scatter3d",
        x=filtered_label1[0],
        y=filtered_label1[1],
        z=filtered_label1[2],
        marker=dict(size=2,
                    color="green")
        )

    scatter3 = dict(
        mode="markers",
        name=parameters[2],
        type="scatter3d",
        x=filtered_label2[0],
        y=filtered_label2[1],
        z=filtered_label2[2],
        marker=dict(size=2,
                    color="blue")
        )

    centroid1 = dict(
        mode="markers",
        name="Centroid " + parameters[0],
        type="scatter3d",
        x=[centroids.iloc[0, 0]],
        y=[centroids.iloc[0, 1]],
        z=[centroids.iloc[0, 2]],
        marker=dict(size=5,
                    color="red"),
        marker_symbol="square",
        )

    centroid2 = dict(
        mode="markers",
        name="Centroid " + parameters[1],
        type="scatter3d",
        x=[centroids.iloc[1,0]],
        y=[centroids.iloc[1,1]],
        z=[centroids.iloc[1,2]],
        marker=dict(size=5,
        color="green"),
        marker_symbol="square"
    )

    centroid3 = dict(
        mode="markers",
        name="Centroid " + parameters[2],
        type="scatter3d",
        x=[centroids.iloc[2,0]],
        y=[centroids.iloc[2,1]],
        z=[centroids.iloc[2,2]],
        marker=dict(size=5,
        color="blue"),
        marker_symbol="square"
    )

    cluster1 = dict(
        name="Salinity (ppt)",
        type="mesh3d",
        x=filtered_label0[0],
        y=filtered_label0[1],
        z=filtered_label0[2],
        color="red",
        opacity=0.1,
        alphahull=5,
        showscale=True
        )

    cluster2 = dict(
        name="Temp C",
        type="mesh3d",
        x=filtered_label1[0],
        y=filtered_label1[1],
        z=filtered_label1[2],
        color="green",
        opacity=0.1,
        alphahull=5,
        showscale=True
        )

    cluster3 = dict(
        name="ODO mg/L",
        type="mesh3d",
        x=filtered_label2[0],
        y=filtered_label2[1],
        z=filtered_label2[2],
        color="blue",
        opacity=0.1,
        alphahull=5,
        showscale=True
        )

    layout = dict(
        title='3d point clustering',
        scene=dict(
            xaxis=dict(zeroline=False),
            yaxis=dict(zeroline=False),
            zaxis=dict(zeroline=False),
            xaxis_title=parameters[0],
            yaxis_title=parameters[1],
            zaxis_title=parameters[2]),
        # title='KMeans Clustering displayed on a 3-dimensional space',
        )

    # scatter = [scatter1, scatter2, scatter3]
    # clusters = [cluster1, cluster2, cluster3]
    my_data = [scatter1, scatter2, scatter3,
               centroid1, centroid2, centroid3,
               cluster1, cluster2, cluster3]

    fig = dict(data=my_data, layout=layout)
    iplot(fig, filename='3d point clustering')

def post_processing(df_training_,label):
    print("Starting post processing:\n",df_training_)
    df_training_['Prediction'] = pd.Series(label)
    # define conditions
    conditions = [df_training_['Target'] == df_training_['Prediction'],
                  df_training_['Target'] != df_training_['Prediction']]

    # define choices
    choices = [0, 1]

    # create new column in DataFrame that displays results of comparisons
    df_training_['winner'] = np.select(conditions, choices)
    print(df_training_)
    return df_training_

def calculate_accuracy(df_training_):
    # view the DataFrame
    not_a_match=df_training_['winner'].sum()
    a = not_a_match/9239
    b = a*100
    accuracy = 100-b
    return accuracy

def plotting_var_correlation(df):
    # -------------------------------Heat map to identify highly correlated variables-------------------------
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(),
                annot=True,
                linewidths=.5,
                center=0,
                cbar=False,
                cmap="YlGnBu")
    plt.show()

def plotting_outliers(df,parameters):
    # --Checking Outliers
    plt.figure(figsize=(15, 5))
    pos = 0
    for i in df.columns:
        ax = plt.subplot(1, 3, pos+1)
        sns.boxplot(data=df[i], orient="h")
        ax.set_title(parameters[pos])
        pos += 1
    plt.show()

if __name__ == '__main__':

    idx_param = 1
    max_clusters = 10
    parameters=[["Salinity (ppt)","Temp C", "pH"],
                ["Salinity (ppt)","Temp C","ODO mg/L"],
                ["Salinity (ppt)","pH","ODO mg/L"],
                ["Temp C","pH","ODO mg/L"]]
    parameter = parameters[idx_param]

    df_all = pd.read_csv('All_Data/biscayne_all_data_filtered.csv')
    print("The descriptive stats for all parameters already filtered: \n",df_all.describe())

    df_train = df_all[parameter]
    print("The descriptive stats for the selected parameters already filtered: \n", df_train.describe())

    print("Heat map to identify highly correlated variables:\n")
    plotting_var_correlation(df_train)

    print("Checking Outliers:\n")
    plotting_outliers(df_train, parameters[idx_param])

    df_norm = standardize(df_train)
    print("Info on the normalized data: \n", df_norm.info())

    """Create KmeansModel:"""
    kmodel = KmeansModel(df_norm, max_clusters)
    print("================================================================================")
    print("distortions: ", len(kmodel.distortions), kmodel.distortions)
    print("inertias: ", len(kmodel.inertias), kmodel.inertias)
    print("dist_map: ", len(kmodel.dist_map), kmodel.dist_map)
    print("iner_map: ", len(kmodel.iner_map), kmodel.iner_map)
    print("K: ", kmodel.K)
    print("Silhouette Coefficient: ", len(kmodel.sil_coef), kmodel.sil_coef)
    kmodel.plotting_k_elbow_method()
    #
    # """Silhouette Coefficient"""
    kmodel.plotting_silhouette()
    print("================================================================================")

    """Applying kmeans:"""
    this_label, centroids = kmeans_method(df_norm)
    print(this_label,len(this_label))

    """Calculating accuracy"""
    df_ = post_processing(df_all, this_label)
    accuracy = calculate_accuracy(df_)
    print("The accuracy of KMeans is: " + str(accuracy) + "%.")

    """Plotting the clusters"""
    df_centroids = convert_centroids_toDF(centroids)
    # plotting_clusters_plotly_with_centroids(df_norm, this_label, df_centroids, parameter)
    # fig = plotting_clusters_plotly(df_norm,this_label,centroids,parameter)
    # iplot(fig, filename='3d point clustering')
    print("Done")
    # TODO: precision, recall and f1 measure for all different combination of 3 parameters
    # Source: https://www.guavus.com/technical-blog/unsupervised-machine-learning-validation-techniques/