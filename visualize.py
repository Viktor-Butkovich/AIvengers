import preprocess
import constants
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

import matplotlib.patches as mpatches

X_train, y_train = preprocess.preprocess('codon_usage.csv', random_state=42, split=False)

def kingdom_to_color(kingdom: str):
    color = {
        'arc': (0, 0, 0),
        'plm': (125, 125, 125),
        'rod': (255, 0, 0),
        'phg': (0, 255, 0),
        'pri': (0, 0, 255),
        'vrt': (0, 125, 125),
        'bct': (125, 0, 125),
        'pln': (125, 125, 0),
        'inv': (0, 255, 255),
        'mam': (255, 0, 255),
        'vrl': (255, 255, 0),
    }[kingdom]
    return(color[0] / 255, color[1] / 255, color[2] / 255)

valid_visualizer = False
prompt = 'Enter a type of visualizer to run: t-SNE, PCA\n'
visualizer_type = input(prompt)
while not valid_visualizer:
    if visualizer_type in ['t-SNE', 'PCA']:
        valid_visualizer = True

        if visualizer_type == 't-SNE':
            X_transformed = TSNE(n_components=2).fit_transform(X_train) # reduce components to 2 dimensions
        elif visualizer_type == 'PCA':
            X_transformed = PCA(n_components=2).fit_transform(X_train) # reduce components to 2 dimensions

        plt.title(visualizer_type)
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=pd.Series(y_train).apply(kingdom_to_color), s=5)
        plt.xlabel("Reduced Dimension 1")
        plt.ylabel("Reduced Dimension 2")

        handles = [mpatches.Patch(color=kingdom_to_color(kingdom), label=kingdom) for kingdom in constants.kingdoms]
        plt.legend(handles, constants.kingdom_names, ncol=1, bbox_to_anchor=(1, 1))

        plt.show()

    else:
        print('That is not a valid visualizer type.\n')
        visualizer_type = input(prompt)
