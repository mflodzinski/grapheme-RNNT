from logging import Logger
import copy
import os
import time

from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from model import Transducer
from optim import Optimizer
from tokenizer import CharTokenizer
from utils import AttrDict
from typing import Optional, Union
import utils


def train(
    epoch: int,
    config: AttrDict,
    model: Transducer,
    train_data: DataLoader,
    optimizer: Optimizer,
    logger: Logger,
    device: Union[str, torch.device],
    tracker=None,
):
    model.train()
    start_epoch_time = time.process_time()
    total_loss = 0
    batch_steps = len(train_data)

    optimizer.epoch()

    for step, (inputs, inputs_length, targets, targets_length) in enumerate(train_data):
        inputs, inputs_length, targets, targets_length = (
            inputs.to(device),
            inputs_length.to(device),
            targets.to(device),
            targets_length.to(device),
        )

        optimizer.zero_grad()
        start_step_time = time.process_time()

        loss = model(inputs, inputs_length, targets, targets_length)
        loss = loss.mean()
        loss.backward()
        total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config.training.max_grad_norm
        )

        optimizer.step()
        avg_loss = total_loss / (step + 1)

        if optimizer.global_step % config.training.show_every == 0:
            end_step_time = time.process_time()
            progress = (step / batch_steps) * 100
            logger.info(
                f"-Training-Epoch:{epoch}({progress:.5f}%), Global Step:{optimizer.global_step}, "
                f"Learning Rate:{optimizer.lr:.6f}, Grad Norm:{grad_norm:.5f}, Loss:{loss.item():.5f}, "
                f"AverageLoss:{avg_loss:.5f}, Run Time:{end_step_time - start_step_time:.3f}"
            )
    if tracker is not None:
        tracker.add_scalar("loss_train", avg_loss, epoch)

    end_epoch_time = time.process_time()
    logger.info(
        f"-Training-Epoch:{epoch}, Average Loss: {avg_loss:.5f}, Epoch Time: {end_epoch_time - start_epoch_time:.3f}"
    )


def eval(
    epoch: int,
    model: Transducer,
    eval_data: DataLoader,
    logger: Logger,
    special_tokens: list[int],
    device: Union[str, torch.device],
    tracker=None,
    split_name: str = "val",
    max_samples: Optional[int] = None,
    log_loss: bool = True,
):
    model.eval()
    total_loss = 0
    total_dist = 0
    total_word = 0
    batch_steps = len(eval_data)
    seen_samples = 0

    with torch.no_grad():
        for step, (inputs, inputs_length, targets, targets_length) in enumerate(eval_data):
            inputs, inputs_length, targets, targets_length = (
                inputs.to(device),
                inputs_length.to(device),
                targets.to(device),
                targets_length.to(device),
            )

            loss = model(inputs, inputs_length, targets, targets_length)
            total_loss += loss.mean().item()

            predictions = model.recognize(inputs, inputs_length)

            transcripts = [
                targets.cpu().numpy()[i][: targets_length[i].item()]
                for i in range(targets.size(0))
            ]

            predictions = utils.remove_special_tokens(predictions, special_tokens)
            transcripts = utils.remove_special_tokens(transcripts, special_tokens)

            dist, num_words = utils.compute_cer(predictions, transcripts)
            total_dist += dist
            total_word += num_words
            seen_samples += inputs.size(0)

            process = step / batch_steps * 100
            cer = total_dist / total_word * 100
            logger.info(
                f"-{split_name.capitalize()}-Epoch:{epoch}({process:.5f}%), CER: {cer:.5f}%"
            )

            if max_samples is not None and seen_samples >= max_samples:
                break

    avg_loss = total_loss / (step + 1)
    cer = total_dist / total_word * 100
    logger.info(
        f"-{split_name.capitalize()}-Epoch:{epoch:4d}, AverageLoss:{avg_loss:.5f}, AverageCER: {cer:.5f}%"
    )

    if tracker is not None:
        if log_loss:
            tracker.add_scalar(f"loss_{split_name}", avg_loss, epoch)
        tracker.add_scalar(f"cer_{split_name}", cer, epoch)

    return avg_loss, cer


def is_improved(metric: float, best_metric: float, min_delta: float):
    return metric < best_metric - min_delta


def train_model(
    config: AttrDict,
    model: Transducer,
    optimizer: Optimizer,
    train_data: DataLoader,
    val_data: DataLoader,
    test_data: DataLoader,
    logger: Logger,
    device: Union[str, torch.device],
    tracker,
    tokenizer: CharTokenizer,
):
    special_tokens = [
        tokenizer.stoi[tokenizer.special_tokens["pad"]],
        tokenizer.stoi[tokenizer.special_tokens["blank"]],
    ]
    early_stopping = bool(config.training.early_stopping)
    patience = config.training.early_stopping_patience or 0
    min_delta = config.training.early_stopping_min_delta or 0.0
    restore_best = (
        True
        if config.training.restore_best_model is None
        else bool(config.training.restore_best_model)
    )
    best_val_loss = float("inf")
    best_epoch = None
    best_state_dict = None
    checks_without_improvement = 0
    eval_every = config.training.eval_every or config.training.save_every
    train_cer_samples = config.training.train_cer_samples or 0

    if early_stopping and not config.training.evaluate:
        logger.info("Early stopping is enabled but evaluation is disabled.")

    for epoch in range(config.training.epochs):
        train(epoch, config, model, train_data, optimizer, logger, device, tracker)

        should_evaluate = config.training.evaluate and epoch % eval_every == 0
        if should_evaluate:
            if train_cer_samples > 0:
                eval(
                    epoch,
                    model,
                    train_data,
                    logger,
                    special_tokens,
                    device,
                    tracker,
                    split_name="train",
                    max_samples=train_cer_samples,
                    log_loss=False,
                )
            val_loss, _ = eval(
                epoch,
                model,
                val_data,
                logger,
                special_tokens,
                device,
                tracker,
                split_name="val",
            )
            eval(
                epoch,
                model,
                test_data,
                logger,
                special_tokens,
                device,
                tracker,
                split_name="test",
            )
            optimizer.step_scheduler(val_loss, logger)

            if early_stopping:
                if is_improved(val_loss, best_val_loss, min_delta):
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(model.state_dict())
                    checks_without_improvement = 0
                    save_name = os.path.join(config.data.exp_name, "best.epoch")
                    utils.save_model(model, save_name)
                    logger.info(
                        f"Epoch {epoch} improved validation loss to {val_loss:.5f}; "
                        f"best model saved to {save_name}."
                    )
                else:
                    checks_without_improvement += 1
                    logger.info(
                        f"Validation loss did not improve for "
                        f"{checks_without_improvement}/{patience} checks."
                    )
                    if checks_without_improvement >= patience:
                        logger.info(
                            f"Early stopping at epoch {epoch}. "
                            f"Best validation loss was {best_val_loss:.5f} at epoch {best_epoch}."
                        )
                        break

        if epoch % config.training.save_every == 0:
            utils.save_model_checkpoint(model, epoch, config, logger)

        utils.adjust_learning_rate(optimizer, epoch, config, logger)

    if early_stopping and restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(
            f"Restored best model from epoch {best_epoch} "
            f"with validation loss {best_val_loss:.5f}."
        )

    logger.info("The training process is OVER!")

    if tracker is not None:
        tracker.finish()


def main():
    CONFIG_PATH = "config/config.yaml"
    #torch.set_float32_matmul_precision("high")

    config = utils.load_config(CONFIG_PATH)
    logger = utils.setup_logger(config)
    tracker = utils.create_tracker(config)
    device = utils.setup_device(logger)

    train_data, test_data, val_data, tokenizer = utils.prepare_data_loaders(config)
    model = utils.initialize_model(config, tokenizer.vocab_size, device)
    optimizer = utils.create_optimizer(model, config.optim)
    utils.log_model_parameters(model, logger)

    train_model(
        config=config,
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        logger=logger,
        device=device,
        tracker=tracker,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
