import os
import plotly.express as px
import sys

sys.path.append('../..')
from shared.utils.uncertainties import plot_uncert_train, plot_uncert_test, plotly_train, plot_uncert_comparative, plotly_test

if __name__ == "__main__":
    smooth = 2
    plot_variance = False
    train_paths = [
        # "uncertainties/train/base.txt",
        "uncertainties/train/bnn2.txt",
        "uncertainties/train/bootstrap.txt",
        "uncertainties/train/dropout.txt",
        "uncertainties/train/sensitivity.txt",
        "uncertainties/train/vae.txt",
        "uncertainties/train/aleatoric.txt",
        "uncertainties/train/bootstrap2.txt",
        "uncertainties/train/dropout2.txt",
    ]
    names = [
        # "Base",
        "Bayesian NN",
        "Bootstrap",
        "Dropout",
        "Sensitivity",
        "VAE",
        "Aleatoric",
        "Bootstrap 2",
        "Dropout 2",
    ]
    multipliers = [1] * 10
    colors_px = px.colors.qualitative.Plotly
    linewidths = [2] * 10

    if not os.path.exists("images"):
        os.makedirs("images")

    plot_uncert_train(
        train_paths, names, colors=colors_px, linewidths=linewidths, smooth=smooth, plot_variance=plot_variance, multipliers=multipliers
    )
    plotly_train(train_paths, names, colors=colors_px, smooth=smooth, plot_variance=plot_variance)

    test_paths = [
        # "uncertainties/test/base.txt",
        "uncertainties/test/bnn2.txt",
        "uncertainties/test/bootstrap.txt",
        "uncertainties/test/dropout.txt",
        "uncertainties/test/sensitivity.txt",
        "uncertainties/test/vae.txt",
        "uncertainties/test/aleatoric.txt",
        "uncertainties/test/bootstrap2.txt",
        "uncertainties/test/dropout2.txt",
    ]

    plot_uncert_test(test_paths, names, colors=colors_px, linewidths=linewidths, smooth=smooth, plot_variance=plot_variance, multipliers=multipliers)
    plotly_test(test_paths, names, colors=colors_px, smooth=smooth, plot_variance=plot_variance)

    plot_uncert_comparative(train_paths, test_paths, names, linewidths)