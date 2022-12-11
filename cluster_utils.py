import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


# combine seasons
def combine_seasons(seasons):
    # load first season of group
    final = pd.read_csv(seasons[0])
    final = final.drop_duplicates(subset='Id', keep="first")

    for season in seasons[1:]:
        df = pd.read_csv(season)

        # first line for each player has their totals for the year
        df = df.drop_duplicates(subset='Id', keep="first")

        # add stats to previous seasons
        final = pd.concat([final, df])

    # group by player ID and get mean stats
    final = final.groupby(['Id']).agg({'Rk': 'first',
                                       'Player': 'first',
                                       'Age': 'mean',
                                       'Tm': 'last',
                                       'Pos': 'last',
                                       'GP': 'mean',
                                       'G': 'mean',
                                       'A': 'mean',
                                       'PTS': 'mean',
                                       '+/-': 'mean',
                                       'PIM': 'mean',
                                       'PS': 'mean',
                                       'EV': 'mean',
                                       'PP': 'mean',
                                       'SH': 'mean',
                                       'GW': 'mean',
                                       'EV.1': 'mean',
                                       'PP.1': 'mean',
                                       'SH.1': 'mean',
                                       'S': 'mean',
                                       'S%': 'mean'})

    final = final.reset_index()

    final = final.sort_values(by=['GP'], ascending=False)
    final = final.head(int(len(final)*(75/100)))
    final = final.sort_values(by=['Rk'], ascending=True)

    return final

# prep seasons
def prep_group(group):
    players = group[['Player', 'Pos', 'Id']]
    X = group.drop(['Player', 'Pos', 'Id', 'Tm'], axis=1)

    # drop bottom X% of players by GP?

    return X, players

# splitting offense and defense
def split_pos(group):
    X = group.copy()

    # players = X[['Player', 'Pos', 'Id']]
    # X = X.drop(['Player', 'Pos', 'Id', 'Tm'], axis=1)

    defense = X[X['Pos'] == 'D']
    d_players = defense[['Player', 'Pos', 'Id']]
    defense = defense.drop(['Player', 'Pos', 'Id', 'Tm'], axis=1)

    offense = X[X['Pos'] != 'D']
    o_players = offense[['Player', 'Pos', 'Id']]
    offense = offense.drop(['Player', 'Pos', 'Id', 'Tm'], axis=1)

    return offense, o_players, defense, d_players

# getting goons
def get_goons(group):
    n = 15 # returns top 15% PIM
    top_pim = group.copy()
    top_pim = top_pim.sort_values(by=['PIM'], ascending=False)
    top_pim = top_pim.head(int(len(group)*(n/100)))

    bot_pts = group.copy()
    bot_pts = bot_pts.sort_values(by=['PTS'], ascending=True)
    bot_pts = bot_pts.head(int(len(group)*(n/100)))

    goons = top_pim.merge(bot_pts, how='inner', on=['Id', 'Rk', 'Player', 'Age',
                                                    'Tm', 'Pos', 'GP', 'G', 'A',
                                                    'PTS', '+/-', 'PIM', 'PS',
                                                    'EV', 'PP', 'SH', 'GW', 'EV.1',
                                                    'PP.1', 'SH.1', 'S', 'S%'])

    return goons

### clustering
# clustering with PCA
def cluster_groups(X, player, goon_df, path=None):
    names = player.Player.values
    range_n_clusters = [2, 3, 4, 5, 6]
    silhouettes = {}

    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        ax1.set_xlim([-0.5, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('scale',StandardScaler()),
            ('pca', PCA(n_components=2,random_state=671)),
            ('cluster', KMeans(n_clusters=n_clusters) ),
        ])
        cluster_labels = pipe.fit_predict(X)
        Xtransformed = pipe.transform(X) # imputed and scaled, fed to pca, and return results
        Xtransformed2 = pd.concat([pd.DataFrame({'who':names}),pd.DataFrame(Xtransformed)],axis=1)

        clusterer = pipe.named_steps.cluster

        # The silhouette_score gives the average value for all the samples.
        silhouette_avg = silhouette_score(Xtransformed, cluster_labels)
        silhouettes[n_clusters] = silhouette_avg
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(Xtransformed, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        ### 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

        x = sns.scatterplot(x=Xtransformed[:, 0], y=Xtransformed[:, 1], marker='o', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        ### Plotting goon names
        # goons = ['Dale Hunter', 'Sean Avery', 'Marty McSorley', 'Bob Propert',
        #          'Rob Ray', 'Craig Berube', 'Tim Hunter', 'Tie Domi', 'Donald Brashear',
        #          'Shane Churla', 'Milan Lucic', 'Tom Wilson']
        goons = list(goon_df['Player'].values)
        for line in range(0, Xtransformed2.shape[0]):
            if Xtransformed2['who'][line] in goons:
                 x.text(Xtransformed2[0][line]+0.01, Xtransformed2[1][line],
                 Xtransformed2['who'][line], horizontalalignment='left',
                 size='medium', color=colors[line])

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        if path:
            plt.savefig('figs/cluster_plots/' + path + '/' + str(n_clusters) + '.png')
        # plt.savefig('figs/cluster_plots/cluster.png')


    return silhouettes

# clustering without PCA
def cluster_no_pca(X, player, goon_df, path=None):
    names = player.Player.values
    range_n_clusters = [2, 3, 4, 5, 6]
    silhouettes = {}

    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        ax1.set_xlim([-0.5, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('scale',StandardScaler()),
        ])

        Xtransformed = pipe.fit_transform(X)
        clustering = KMeans(n_clusters=n_clusters, random_state=671)
        clustering.fit(Xtransformed)

        cluster_labels = clustering.predict(Xtransformed)

        # imputed and scaled, fed to pca, and return results
        Xtransformed2 = pd.concat([pd.DataFrame({'who':names}),pd.DataFrame(Xtransformed)],axis=1)

        # return Xtransformed
        clusterer = clustering

        # The silhouette_score gives the average value for all the samples.
        silhouette_avg = silhouette_score(Xtransformed, cluster_labels)
        silhouettes[n_clusters] = silhouette_avg
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(Xtransformed, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        ### 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

        x = sns.scatterplot(x=Xtransformed[:, 5], y=Xtransformed[:, 7], marker='o', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        ### plotting goon names
        # goons = ['Dale Hunter', 'Sean Avery', 'Marty McSorley', 'Bob Propert',
        #          'Rob Ray', 'Craig Berube', 'Tim Hunter', 'Tie Domi', 'Donald Brashear',
        #          'Shane Churla', 'Milan Lucic', 'Tom Wilson']
        goons = goons = list(goon_df['Player'].values)
        for line in range(0, Xtransformed2.shape[0]):
            if Xtransformed2['who'][line] in goons:
                 x.text(Xtransformed2[5][line], Xtransformed2[7][line],
                     Xtransformed2['who'][line], horizontalalignment='left',
                     size='small', color=colors[line])

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Points")
        ax2.set_ylabel("Penalty Minutes")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        if path:
            plt.savefig('figs/cluster_plots/' + path + '/' + str(n_clusters) + 'pca.png')

    return silhouettes

# final clustering
def final_cluster(X, player, goon_df, n_clusters=2, path=None):
    names = player.Player.values
    range_n_clusters = [2, 3, 4, 5, 6]
    silhouettes = {}

    # for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    ax1.set_xlim([-0.5, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scale',StandardScaler()),
    ])

    Xtransformed = pipe.fit_transform(X)
    clustering = KMeans(n_clusters=n_clusters, random_state=671)
    clustering.fit(Xtransformed)

    cluster_labels = clustering.predict(Xtransformed)

    # imputed and scaled, fed to pca, and return results
    Xtransformed2 = pd.concat([pd.DataFrame({'who':names}),pd.DataFrame(Xtransformed)],axis=1)

    # return Xtransformed
    clusterer = clustering

    # The silhouette_score gives the average value for all the samples.
    silhouette_avg = silhouette_score(Xtransformed, cluster_labels)
    silhouettes[n_clusters] = silhouette_avg
    print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(Xtransformed, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    ### 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

    x = sns.scatterplot(x=Xtransformed[:, 5], y=Xtransformed[:, 7], marker='o', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    ### plotting goon names
    goons2 = ['Dale Hunter', 'Sean Avery', 'Marty McSorley', 'Bob Propert',
             'Rob Ray', 'Craig Berube', 'Tim Hunter', 'Tie Domi', 'Donald Brashear',
             'Shane Churla', 'Milan Lucic', 'Tom Wilson']
    goons = goons = list(goon_df['Player'].values)
    for line in range(0, Xtransformed2.shape[0]):
        if Xtransformed2['who'][line] in goons:
                x.text(Xtransformed2[5][line], Xtransformed2[7][line],
                    Xtransformed2['who'][line], horizontalalignment='left',
                    size='small', color=colors[line])#colors[line])
        if Xtransformed2['who'][line] in goons2:
                x.text(Xtransformed2[5][line], Xtransformed2[7][line],
                    Xtransformed2['who'][line], horizontalalignment='left',
                    size='small', color='orange')#colors[line])

    if player.iloc[0].Pos =='D':
        ax2.set_title("Plot of Defensemen")
    else:
        ax2.set_title("Plot of Forwards")
    ax2.set_xlabel("Points")
    ax2.set_ylabel("Penalty Minutes")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')

    if path:
        plt.savefig('figs/cluster_plots/final_clusters/' + path + '.png')

    return cluster_labels, silhouettes

### Scree plots
# without PCA
def scree_plot(X, path=None):
    scores = []
    for n_clusters in range(1, 6):
        pipe = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scale',StandardScaler()),
        ])

        Xtransformed = pipe.fit_transform(X)

        kmeans = KMeans(n_clusters = n_clusters)
        kmeans.fit(Xtransformed)
        scores.append(-kmeans.score(Xtransformed))

    plt.scatter(list(range(1, 6)), scores)
    plt.xticks([1,2,3,4,5,6])

    if path:
         plt.savefig('figs/scree_plots/' + path)

    plt.show()

    return scores

# with PCA
def scree_plot_pca(X, path=None):
    scores = []
    for n_clusters in range(1, 6):
        pipe = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scale',StandardScaler()),
        ('pca', PCA(n_components=2,random_state=671)),
        ])

        Xtransformed = pipe.fit_transform(X)

        kmeans = KMeans(n_clusters = n_clusters)
        kmeans.fit(Xtransformed)
        scores.append(-kmeans.score(Xtransformed))

    plt.scatter(list(range(1, 6)), scores)
    plt.xticks([1,2,3,4,5,6])

    if path:
         plt.savefig('figs/scree_plots/' + path)

    plt.show()

    return scores
