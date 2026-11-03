"""🏷️ Torch-native multi-label training and evaluation loops."""

import warnings
from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from skald import Logger
from torch.utils.data import DataLoader
from torchmetrics import ClasswiseWrapper, MetricCollection, classification

from metrics import CardinalityConfusionMatrix, MultilabelConfusionMatrix
from visualizations import plot_cardinality_confusion_matrix, plot_multilabel_confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")


def build_multilabel_metrics(
    num_labels: int,
    class_names: list[str],
    include_classwise_metrics: bool = False,
) -> MetricCollection:
    """📏 Build the multi-label metric collection used for validation and test."""
    metrics = [
        MetricCollection(
            {
                "Precision": classification.MultilabelPrecision(num_labels=num_labels, average="macro"),
                "Recall": classification.MultilabelRecall(num_labels=num_labels, average="macro"),
                "mAP": classification.MultilabelAveragePrecision(num_labels=num_labels, average="macro"),
            },
            postfix="|*",
        ),
    ]
    if include_classwise_metrics:
        metrics.append(
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
        )
    return MetricCollection(metrics)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    augmentation: Callable[[torch.Tensor], torch.Tensor],
    loss_function: torch.nn.Module,
    device: torch.device,
    stream: str,
    task: int,
    epoch: int | None,
    logger: Logger,
    class_names: list[str],
    include_classwise_metrics: bool = False,
    save_confusion_artifacts: bool = False,
) -> tuple[float, dict[str, float]]:
    model.eval()
    metrics = build_multilabel_metrics(
        num_labels=len(class_names),
        class_names=class_names,
        include_classwise_metrics=include_classwise_metrics,
    ).to(device)
    if save_confusion_artifacts:
        cmat = MultilabelConfusionMatrix(len(class_names)).to(device)
        ccmat = CardinalityConfusionMatrix(len(class_names)).to(device)

    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch_images, batch_targets in dataloader:
            images = batch_images.to(device, non_blocking=True)
            targets = batch_targets.to(device, dtype=torch.float32, non_blocking=True)
            images = augmentation(images)

            predictions = model(images)
            loss = loss_function(predictions, targets)

            total_loss += float(loss.item())
            total_batches += 1
            metrics(predictions, targets.int())
            if save_confusion_artifacts:
                cmat(predictions, targets.int())
                ccmat(predictions, targets.int())

    # 💹 log loss
    loss = round(total_loss / max(total_batches, 1), 4)
    logger.log_metric("loss", loss, stream=stream, epoch=epoch, task=task)
    # 📃 log metrics
    metrics_dict = {name: round(float(value.item()), 4) for name, value in metrics.compute().items()}
    for metric_name, value in metrics_dict.items():
        name, cls = metric_name.split("|")
        logger.log_metric(
            name,
            value,
            task=task,
            stream=stream,
            epoch=epoch,
            pattern=cls,
        )

    if save_confusion_artifacts:
        cmat = cmat.compute()
        if epoch is None:
            fig = plot_multilabel_confusion_matrix(cmat)
            torch.save(
                cmat,
                logger.artifacts_dir / f"task={task} stream={stream} cmat.pt",
            )
            fig.savefig(logger.artifacts_dir / f"task={task} stream={stream} cmat.png")
            plt.close(fig)

        ccmat = ccmat.compute()
        if epoch is None:
            fig = plot_cardinality_confusion_matrix(ccmat)
            torch.save(
                ccmat,
                logger.artifacts_dir / f"task={task} stream={stream} ccmat.pt",
            )
            fig.savefig(logger.artifacts_dir / f"task={task} stream={stream} cardinality confusion.png")
            plt.close(fig)

    model.train()

    return loss, metrics_dict


def fit(
    method,
    task_context,
    optimizer,
    lr_scheduler,
    epochs: int,
    logger,
    step_mode: str = "step",
) -> list[dict[str, float]]:
    model = task_context.model
    model.train()
    train_loader = task_context.build_train_loader(task_context.stream.train[task_context.task])
    val_loader = task_context.build_val_loader(task_context.stream.test["present"][task_context.task])
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = torch.zeros((), device=task_context.device)
        batch_count = 0
        for batch_images, batch_targets in train_loader:
            images = batch_images.to(task_context.device, non_blocking=True)
            targets = batch_targets.to(task_context.device, dtype=torch.float32, non_blocking=True)
            images = task_context.augmentation(images)

            losses = method.training_step(task_context, images, targets, optimizer)
            for loss in losses:
                if step_mode == "step" and lr_scheduler is not None:
                    lr_scheduler.step()
                task_context.step += 1
                epoch_loss += loss
                batch_count += 1

                logger.log(
                    "lr",
                    float(optimizer.param_groups[0]["lr"]),
                    step=task_context.step,
                    epoch=epoch,
                    task=task_context.task,
                )

        val_loss, metrics = evaluate(
            model,
            dataloader=val_loader,
            augmentation=task_context.augmentation,
            loss_function=task_context.loss_function,
            device=task_context.device,
            stream="present",
            task=task_context.task,
            epoch=epoch,
            logger=logger,
            class_names=task_context.stream.class_names,
        )

        if step_mode == "epoch" and lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        train_loss = float((epoch_loss / max(batch_count, 1)).item())
        logger.log("loss", train_loss, stream="train", epoch=epoch, task=task_context.task)
        lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(lr),
                **metrics,
            },
        )
        print(
            f"epoch {epoch + 1:>4}/{epochs:<4} | train loss={train_loss:>8.4f} | "
            f"val loss={val_loss:>8.4f} | mAP={metrics['mAP|*']:>7.4f} | "
            f"pre={metrics['Precision|*']:>7.4f} | rec={metrics['Recall|*']:>7.4f} | lr={lr:>9.6f}",
        )
    return history
