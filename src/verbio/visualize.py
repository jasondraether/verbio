import seaborn as sns
import matplotlib.pyplot as plt
from verbio import utils


def plot_matrix(df, hue_key: str, feature_keys=None):
    """

    :param df:
    :param hue_key:
    :param feature_keys:
    :return:
    """
    if feature_keys is None:
        feature_keys = utils.get_df_keys(df)
        if hue_key in feature_keys:
            feature_keys.remove(hue_key)
    plotting_keys = [hue_key] + feature_keys
    sns.pairplot(df[plotting_keys], hue=hue_key, plot_kws=dict(marker="+", linewidth=1))
    plt.show()
