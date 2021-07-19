import pandas as pd
import numpy as np
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
from sklearn import preprocessing
from sklearn.cluster import KMeans

def standardize(df_training_):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_training_)
    df_normalized = pd.DataFrame(x_scaled)
    # print(df_normalized.shape)
    return df_normalized

def kmeans_method(df_normalized):
    # Trying fit_predict
    kmeans = KMeans(n_clusters=3,random_state=134)
    label = kmeans.fit_predict(df_normalized)

    centroids = kmeans.cluster_centers_
    return label, centroids

def plotting_clusters_plotly(df_normalized,label,centroids,parameters):
    filtered_label0 = df_normalized[label == 0]
    filtered_label1 = df_normalized[label == 1]
    filtered_label2 = df_normalized[label == 2]

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

if __name__ == '__main__':

    parameters=[["Salinity (ppt)","Temp C", "pH"],
                ["Salinity (ppt)","Temp C","ODO mg/L"],
                ["Salinity (ppt)","pH","ODO mg/L"],
                ["Temp C","pH","ODO mg/L"]]
    parameter = parameters[1]
    df_all = pd.read_csv('All_Data/biscayne_all_data_filtered.csv')
    print("The descriptive stats for all parameters already filtered: \n",df_all.describe())
    df_training_ = df_all[parameter]
    print("The descriptive stats for the selected parameters already filtered: \n",df_training_.describe())
    df_normalized_ = standardize(df_training_)
    print("Info on the normalized data: \n",df_normalized_.info())
    """Applying kmeans:"""
    this_label, centroids = kmeans_method(df_normalized_)
    print(this_label,len(this_label))
    """Calculating accuracy"""
    df_ = post_processing(df_all, this_label)
    accuracy = calculate_accuracy(df_)
    print("The accuracy of KMeans is: " + str(accuracy) + "%.")
    """Plotting the clusters"""
    # fig = plotting_clusters_plotly(df_normalized,this_label,centroids,parameter)
    # iplot(fig, filename='3d point clustering')
    print("Done")