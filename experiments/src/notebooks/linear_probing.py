import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import torch
    from torch.utils.data import DataLoader
    from torch.nn import BCEWithLogitsLoss
    import torch.nn as nn
    import torch.optim as optim
    from torchmetrics import ClasswiseWrapper, MetricCollection, classification
    from rich import print

    from experiments.utils import seed_everything
    from experiments.visualizations import set_paper_style
    from experiments.data import (
        MixedWM38,
        build_single_task_stream,
        evaluation_output_path,
    )
    from endlesswm.io.datasets import H5WM

    set_paper_style()
    return (
        BCEWithLogitsLoss,
        ClasswiseWrapper,
        DataLoader,
        H5WM,
        MetricCollection,
        MixedWM38,
        Path,
        build_single_task_stream,
        evaluation_output_path,
        classification,
        mo,
        nn,
        optim,
        print,
        seed_everything,
        torch,
    )


@app.cell
def _(mo):
    setting = mo.ui.dropdown(["iii_to_mixedwm38", "mixedwm38_to_iii"], value="iii_to_mixedwm38")
    setting
    return (setting,)


@app.cell
def _(print, setting, torch):
    if setting.value == "iii_to_mixedwm38":
        print("load iii")
        model = torch.load(
            r"W:\EndlessWM\experiments\reports\static_complexity\III_f1.0_s42\artifacts\model.pt",
            map_location="cpu",
        ).eval()
    elif setting.value == "mixedwm38_to_iii":
        print("load mixedwm")
        model = torch.load(
            r"W:\EndlessWM\experiments\reports\mixedwm38\f1.0_s42_2026-05-26_14-29-54_595635\artifacts\model.pt",
            map_location="cpu",
        ).eval()
    model
    return (model,)


@app.cell
def _(
    H5WM,
    MixedWM38,
    Path,
    build_single_task_stream,
    evaluation_output_path,
    print,
    seed_everything,
    setting,
    v2,
):
    seed_everything(42)

    if setting.value == "iii_to_mixedwm38":
        print("load mixedwm38 data")

        class_names = [
            "Center",
            "Donut",
            "Edge_Loc",
            "Edge_Ring",
            "Loc",
            "Near_Full",
            "Scratch",
            "Random",
        ]

        stream = build_single_task_stream(
            train_dataset=MixedWM38("experiments/data/downloaded/mixedwm38/Wafer_Map_Datasets.npz"),
            class_names=class_names,
            train_val_split=0.8,
            data_fraction=None,
            seed=42,
        )
    elif setting.value == "mixedwm38_to_iii":
        print("load iii data")
        from endlesswm.patterns import classes as class_names
        from experiments.data import FillNaN

        train_path = Path("experiments/data/published/iii.hdf5")
        transforms = {"sample": v2.Compose([FillNaN(), v2.Resize((52, 52))])}
        train_dataset = H5WM(str(train_path), task=0, transform=transforms)

        val_path = evaluation_output_path(train_path)
        val_dataset = H5WM(str(val_path), task=0, transform=transforms)

        stream = build_single_task_stream(
            train_dataset=train_dataset,
            class_names=class_names,
            val_dataset=val_dataset,
        )

    stream
    return class_names, stream


@app.cell
def _():
    from torchvision.transforms import v2
    from experiments.augmentations import StandardizeEachSample

    augment = v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.RandomRotation(degrees=180)]),
            StandardizeEachSample(),
        ]
    )
    return augment, v2


@app.cell
def _(augment, stream):
    from endlesswm.visualizations import heatmap

    x = augment(stream.train[0][0][0].unsqueeze(0))

    heatmap(x.squeeze().numpy())
    return


@app.cell
def _(mo):
    run_probing = mo.ui.run_button()
    run_probing
    return (run_probing,)


@app.cell
def _(
    BCEWithLogitsLoss,
    ClasswiseWrapper,
    DataLoader,
    MetricCollection,
    augment,
    class_names,
    classification,
    mo,
    model,
    nn,
    optim,
    print,
    run_probing,
    stream,
    torch,
):
    if run_probing.value:
        train_loader = DataLoader(stream.train[0], 64, shuffle=True)
        val_loader = DataLoader(stream.test["present"][0], 64, shuffle=False)

        in_features = 512
        probe = nn.Linear(in_features, 8)

        optimizer = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = BCEWithLogitsLoss()

        num_labels = len(class_names)
        metrics = MetricCollection(
            [
                MetricCollection(
                    {
                        "Precision": classification.MultilabelPrecision(num_labels=num_labels),
                        "Recall": classification.MultilabelRecall(num_labels=num_labels),
                        "mAP": classification.MultilabelAveragePrecision(num_labels=num_labels),
                    },
                    postfix="|*",
                ),
                MetricCollection(
                    {
                        "Precision": ClasswiseWrapper(
                            classification.MultilabelPrecision(num_labels=num_labels, average=None),
                            labels=class_names,
                            prefix="Precision|",
                        ),
                        "Recall": ClasswiseWrapper(
                            classification.MultilabelRecall(num_labels=num_labels, average=None),
                            labels=class_names,
                            prefix="Recall|",
                        ),
                        "mAP": ClasswiseWrapper(
                            classification.MultilabelAveragePrecision(num_labels=num_labels, average=None),
                            labels=class_names,
                            prefix="mAP|",
                        ),
                    },
                ),
            ]
        )

        num_epochs = 20

        for epoch in mo.status.progress_bar(range(num_epochs)):
            # ----- Collect frozen features for train -----
            model.eval()
            train_feats = []
            train_targets = []

            with torch.no_grad():
                for inputs, targets in train_loader:
                    inputs = augment(inputs)

                    feats = model.forward_features(inputs)
                    if feats.dim() == 4:
                        feats = model.global_pool(feats)
                        if feats.dim() == 4:
                            feats = feats.flatten(1)
                        else:
                            feats = feats.squeeze(-1).squeeze(-1)
                    else:
                        feats = feats.flatten(1)

                    train_feats.append(feats)
                    train_targets.append(targets)

            train_feats = torch.cat(train_feats, dim=0)
            train_targets = torch.cat(train_targets, dim=0)

            # ----- Train probe -----
            probe.train()
            optimizer.zero_grad(set_to_none=True)

            train_logits = probe(train_feats)
            train_loss = criterion(train_logits, train_targets)
            train_loss.backward()
            optimizer.step()

            print(f"epoch {epoch + 1:>4}/{num_epochs:<4} | train loss={float(train_loss.item()):>8.4f}")

        # ----- Collect frozen features for val -----
        val_feats = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = augment(inputs)

                feats = model.forward_features(inputs)
                if feats.dim() == 4:
                    feats = model.global_pool(feats)
                    if feats.dim() == 4:
                        feats = feats.flatten(1)
                    else:
                        feats = feats.squeeze(-1).squeeze(-1)
                else:
                    feats = feats.flatten(1)

                val_feats.append(feats)
                val_targets.append(targets)

        val_feats = torch.cat(val_feats, dim=0)
        val_targets = torch.cat(val_targets, dim=0)

        # ----- Validate probe -----
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_feats)
            val_loss = criterion(val_logits, val_targets)

        metrics.update(val_logits, val_targets.int())
        metrics_dict = {name: round(float(value.item()), 4) for name, value in metrics.compute().items()}
        print(metrics_dict)
    return


if __name__ == "__main__":
    app.run()
