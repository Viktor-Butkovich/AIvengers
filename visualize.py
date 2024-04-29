from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.model_selection import train_test_split

kingdoms = ['arc', 'plm', 'rod', 'phg', 'pri', 'vrt', 'bct', 'pln', 'inv', 'mam', 'vrl']
kingdom_names = ['archaea', 'plasmid', 'rodent', 'bacteriophage', 'primate', 'vertebrate', 'bacteria', 'plant', 'invertebrate', 'mammal', 'virus']

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
        'vrl': (255, 165, 0),
    }[kingdom]
    return(color[0] / 255, color[1] / 255, color[2] / 255)

def visualize(visualizer, title: str, data = 'all'):
    if data == 'all':
        X, y = preprocess('codon_usage.csv', random_state=42, split=False)
    else:
        X, y = data
    X_transformed =  visualizer.fit_transform(X)
    plt.title(title)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=pd.Series(y).apply(kingdom_to_color), s=3)
    plt.xlabel("Reduced Dimension 1")
    plt.ylabel("Reduced Dimension 2")

    handles = [mpatches.Patch(color=kingdom_to_color(kingdom), label=kingdom) for kingdom in kingdoms]
    plt.legend(handles, kingdom_names, ncol=1, bbox_to_anchor=(1, 1))

    plt.show()

def preprocess(file_path: str, test_size: float = 0.2, random_state: int = 42, split: bool = True) -> tuple:
    # Load the data
    codon_usage_df = pd.read_csv(file_path, low_memory=False)

    # Extracting the codon frequency columns and converting to numeric, coercing errors to NaN
    codon_columns = codon_usage_df.columns[5:]
    X = codon_usage_df[codon_columns].apply(pd.to_numeric, errors='coerce')
    y = codon_usage_df['Kingdom']

    # Drop rows with any NaN values in X and filter y accordingly
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]

    scaler = StandardScaler()

    if split:
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size, random_state=random_state)
    else:
        X_train, y_train = X_clean, y_clean

    # Standardize the data
    X_train_scaled = scaler.fit_transform(X_train)

    if split:
        X_test_scaled = scaler.transform(X_test)
    
    if split:
        return(X_train_scaled, X_test_scaled, y_train, y_test)
    else:
        return(X_train_scaled, y_train)

visualize(PCA(n_components=2), 'PCA')
visualize(TSNE(n_components=2), 't-SNE')

# Standalone visualize program allows fullscreen for greater detail, which is not automatically available in a Jupyter Notebook
