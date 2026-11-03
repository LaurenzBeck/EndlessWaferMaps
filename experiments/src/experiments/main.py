"""🚂 Single entry point for offline and task-stream experiments."""

import contextlib
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import matplotlib as mpl
import torch
from hydra.utils import instantiate
from loguru import logger as log
from methods import TaskContext
from omegaconf import DictConfig, OmegaConf
from rich import traceback, print
from skald import Logger
from torch.utils.data import DataLoader
from training import fit, evaluate
from utils import seed_everything, _print_section, _print_test_summary
from visualizations import plot_epoch_learning_rate, plot_epoch_losses, plot_epoch_metrics

mpl.use("Agg", force=True)
traceback.install(show_locals=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
@log.catch
def main(config: DictConfig) -> None:
    started_at = time.time()
    output_dir = Path.cwd()

    with (
        Logger(
            dir=output_dir,
            run_name=f"{config.comment}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}",
            metrics_file_format="csv",
            persistence_strategy="lazy",
        ) as logger,
        contextlib.redirect_stderr(sys.stdout),
    ):
        output_dir = Path(logger.run_dir)
        os.chdir(output_dir)

        logger.log_params(OmegaConf.to_object(config))
        logger.debug(f"⚙️ Configuration:\n{OmegaConf.to_yaml(config)}")

        seed_everything(int(config.seed))

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"🖥️ Device: {device}")

        stream = instantiate(
            config.data.stream,
        )
        logger.info(f"⚗️ testing will be performed on streams: {list(stream.test.keys())}")

        model = instantiate(config.model.model).to(device)
        loss_function = instantiate(config.loss.obj).to(device)
        augmentation = instantiate(config.augmentation.obj)

        method = instantiate(config.method.obj)

        task_context = TaskContext(
            model=model,
            stream=stream,
            device=device,
            build_train_loader=partial(
                DataLoader,
                batch_size=int(config.data.batch_size),
                shuffle=True,
                num_workers=int(config.data.num_workers),
                persistent_workers=bool(config.data.persistent_workers),
                pin_memory=device.type == "cuda",
            ),
            build_val_loader=partial(
                DataLoader,
                batch_size=int(config.data.batch_size),
                shuffle=False,
                num_workers=int(config.data.num_workers),
                pin_memory=device.type == "cuda",
            ),
            augmentation=augmentation,
            loss_function=loss_function,
        )

        method.before_training(task_context)
        test_results: dict[str, dict[int, dict[str, float]]] = {}
        for task_index in range(len(stream.train)):
            task_context.task = task_index
            method.before_task(task_context)
            train_loader = task_context.build_train_loader(stream.train[task_index])
            task_count = len(stream.train)
            sample_count = len(train_loader.dataset)
            batch_count = len(train_loader)
            _print_section(
                f"Task {task_index + 1}/{task_count} with {sample_count} training samples -> {batch_count} batches/epoch",
            )
            optimizer = hydra.utils.instantiate(config.optimizer.obj, params=model.parameters())
            if config.scheduler.requires_total_steps:
                total_steps = method.training_updates_per_epoch(task_context, train_loader) * int(config.epochs)
                lr_scheduler = hydra.utils.instantiate(config.scheduler.obj, optimizer=optimizer, total_steps=total_steps)
            else:
                lr_scheduler = hydra.utils.instantiate(config.scheduler.obj, optimizer=optimizer)

            training_history = fit(
                method,
                task_context,
                optimizer,
                lr_scheduler,
                epochs=int(config.epochs),
                logger=logger,
                step_mode=config.scheduler.step_mode,
            )

            plot_epoch_learning_rate(training_history)
            plot_epoch_losses(training_history)
            plot_epoch_metrics(training_history)

            for stream_name in stream.test:
                test_dataset = stream.test[stream_name][task_index]
                if test_dataset is not None:
                    test_loader = task_context.build_val_loader(test_dataset)
                    logger.info(f"📈 evaluate on {stream_name} stream with {len(test_loader.dataset)} samples")
                    test_loss, test_metrics = evaluate(
                        model,
                        dataloader=test_loader,
                        augmentation=task_context.augmentation,
                        loss_function=task_context.loss_function,
                        device=task_context.device,
                        stream=stream_name,
                        task=task_context.task,
                        epoch=None,
                        logger=logger,
                        class_names=task_context.stream.class_names,
                        include_classwise_metrics=bool(config.include_classwise_metrics),
                        save_confusion_artifacts=bool(config.save_confusion_artifacts),
                    )
                    test_results.setdefault(stream_name, {})[task_index] = {"loss": test_loss, **test_metrics}

            method.after_task(task_context)

        _print_test_summary(test_results, len(stream.train))

        elapsed_hours = (time.time() - started_at) / 3600.0
        if bool(getattr(config, "save_model", False)):
            final_model_path = output_dir / "artifacts" / "model.pt"
            torch.save(model, final_model_path)
            logger.info(f"💾 Saved final model: {final_model_path.name}")
        logger.log_metric("total-runtime", elapsed_hours)
        logger.success(f"[Elapsed time = {elapsed_hours:.2f} h]")
        logger.save()


if __name__ == "__main__":
    main()
