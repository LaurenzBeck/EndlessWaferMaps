import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # feature extraction

    The cells below inspect classical feature blocks on representative single-label wafer maps from `iii`.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.feature import hog
    from skimage.transform import warp_polar

    from endlesswm.io.datasets import H5WM
    from endlesswm.patterns import classes as class_names
    from endlesswm.visualizations import heatmap
    from experiments.data import FillNaN

    return FillNaN, H5WM, class_names, heatmap, hog, mo, np, plt, warp_polar


@app.cell
def _(FillNaN, H5WM, class_names, mo):
    dataset = H5WM(
        "experiments/data/published/iii.hdf5",
        transform={"sample": FillNaN()},
    )
    pattern = mo.ui.dropdown(class_names, value=class_names[0], label="pattern")
    pattern
    return dataset, pattern


@app.cell
def _(class_names, dataset):
    labels = dataset.labels
    single_label = (labels.sum(dim=1) == 1).nonzero(as_tuple=True)[0]
    samples = [None] * labels.shape[1]

    for index in single_label.tolist():
        label = int(labels[index].argmax().item())
        if samples[label] is None:
            samples[label] = dataset[index][0]
        if all(sample is not None for sample in samples):
            break

    samples = {class_names[index]: samples[index] for index in range(len(class_names))}
    return (samples,)


@app.cell
def _(heatmap, mo, pattern, samples, warp_polar):
    sample = samples[pattern.value].squeeze().numpy()
    mo.hstack(
        [
            heatmap(sample, title=pattern.value),
            heatmap(warp_polar(sample, radius=None, output_shape=(52, 52)), title="polar projection"),
        ]
    )
    return (sample,)


@app.cell
def _(hog, np, warp_polar):
    def histogram_features(image: np.ndarray, bins: int = 16) -> np.ndarray:
        lo, hi = float(image.min()), float(image.max())
        hi = hi if hi > lo else lo + 1e-6
        hist, _ = np.histogram(image, bins=bins, range=(lo, hi), density=True)
        return hist.astype(np.float32)

    def summary_stats_features(image: np.ndarray) -> np.ndarray:
        q = np.quantile(image, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]).astype(np.float32)
        values = [image.mean(), image.std(), image.min(), image.max(), q[5] - q[3]]
        return np.concatenate([np.asarray(values, dtype=np.float32), q])

    def fft_radial_features(image: np.ndarray, bins: int = 16) -> np.ndarray:
        power = np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 2
        rows, cols = image.shape
        yy, xx = np.indices(image.shape)
        radius = np.hypot(yy - (rows - 1) / 2, xx - (cols - 1) / 2)
        edges = np.linspace(0, radius.max(), bins + 1)
        features = np.array(
            [power[(radius >= left) & (radius < right)].mean() for left, right in zip(edges[:-1], edges[1:])],
            dtype=np.float32,
        )
        features = np.nan_to_num(np.log1p(features), nan=0.0)
        return features / features.sum() if features.sum() > 0 else features

    def polar_projection_features(image: np.ndarray, rows: int = 16, cols: int = 16) -> np.ndarray:
        polar = warp_polar(image, radius=None, output_shape=(rows, cols)).astype(np.float32)
        radial = polar.mean(axis=1)
        angular = polar.mean(axis=0)
        radial = (radial - radial.mean()) / (radial.std() + 1e-6)
        angular = (angular - angular.mean()) / (angular.std() + 1e-6)
        return np.concatenate([radial, angular]).astype(np.float32)

    def hog_features(image: np.ndarray) -> np.ndarray:
        return hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
            feature_vector=True,
        ).astype(np.float32)

    def combine_features(image: np.ndarray, builders) -> np.ndarray:
        return np.concatenate([build(image) for build in builders])

    return (
        combine_features,
        fft_radial_features,
        histogram_features,
        hog_features,
        polar_projection_features,
        summary_stats_features,
    )


@app.cell
def _(np, plt):
    def plot_features(values, title: str):
        values = np.asarray(values).ravel()
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(np.arange(len(values)), values, width=1.0)
        ax.set_title(title)
        ax.set_xlabel("feature index")
        ax.set_ylabel("value")
        fig.tight_layout()
        return fig

    return (plot_features,)


@app.cell
def _(
    combine_features,
    fft_radial_features,
    histogram_features,
    hog_features,
    mo,
    plot_features,
    polar_projection_features,
    sample,
    summary_stats_features,
):
    mo.vstack(
        [
            plot_features(histogram_features(sample), "histogram"),
            plot_features(summary_stats_features(sample), "summary statistics"),
            plot_features(fft_radial_features(sample), "radially averaged FFT"),
            plot_features(polar_projection_features(sample), "polar projection"),
            plot_features(hog_features(sample), "HOG"),
            plot_features(combine_features(sample, [summary_stats_features, fft_radial_features, polar_projection_features]), "combined"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
