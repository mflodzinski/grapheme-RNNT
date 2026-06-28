from pathlib import Path
from typing import Any, Union
from logging import Logger
import logging
import os

import yaml
import torch
import editdistance

from data import build_data_loader
from optim import Optimizer
from model import Transducer
from tokenizer import CharTokenizer


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != "":
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def compute_cer(preds, labels):
    dist = sum(editdistance.eval(label, pred) for label, pred in zip(labels, preds))
    total = sum(len(l) for l in labels)
    return dist, total


def count_parameters(model: Transducer):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if "encoder" in name:
            enc += param.nelement()
        elif "decoder" in name:
            dec += param.nelement()
    return n_params, enc, dec


def init_parameters(model: Transducer, type: str = "uniform"):
    for p in model.parameters():
        if p.dim() >= 2:
            if type == "xnoraml":
                torch.nn.init.xavier_normal_(p)
            elif type == "uniform":
                torch.nn.init.uniform_(p, -0.1, 0.1)


def save_model(model: Transducer, save_name: str):
    checkpoint = {
        "encoder": (model.encoder.state_dict()),
        "decoder": (model.decoder.state_dict()),
        "joint": (model.joint.state_dict()),
    }
    torch.save(checkpoint, save_name)


def remove_special_tokens(sequences: list[list[int]], special_tokens: list[int]):
    return [
        [token for token in sequence if token not in special_tokens]
        for sequence in sequences
    ]


def load_config(config_path: Union[Path, str]):
    with open(config_path) as file:
        config = AttrDict(yaml.load(file, Loader=yaml.FullLoader))
    return config


def setup_logger(config: AttrDict):
    return init_logger(os.path.join(config.data.exp_name, "logger"))


def setup_device(logger: Logger):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    logger.info(f"# the device is: {device}")
    return torch.device(device)


def log_model_parameters(model: Transducer, logger: Logger):
    n_params, enc, dec = count_parameters(model)
    logger.info(f"# the number of parameters in the Model: {n_params}")
    logger.info(f"# the number of parameters in the Encoder: {enc}")
    logger.info(f"# the number of parameters in the Decoder: {dec}")
    logger.info(f"# the number of parameters in the JointNet: {n_params - enc - dec}")


def to_plain_dict(value: Any):
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    return value


class WandbTracker:
    def __init__(self, config: AttrDict):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases tracking is enabled. Install wandb or set "
                "`wandb.enabled: False` in config/config.yaml."
            ) from exc

        wandb_config = config.wandb or AttrDict()
        self.run = wandb.init(
            project=wandb_config.project or config.data.exp_name,
            entity=wandb_config.entity,
            name=wandb_config.name,
            mode=wandb_config.mode or "online",
            config=to_plain_dict(config),
        )

    def add_scalar(self, name: str, value: float, step: int):
        self.run.log({name: value}, step=step)

    def finish(self):
        self.run.finish()


def create_tracker(config: AttrDict):
    wandb_config = config.wandb or AttrDict()
    if wandb_config.enabled:
        return WandbTracker(config)
    return None


def save_model_checkpoint(
    model: Transducer, epoch: int, config: AttrDict, logger: Logger
):
    save_name = os.path.join(config.data.exp_name, f"{epoch}.epoch")
    save_model(model, save_name)
    logger.info(f"Epoch {epoch} model has been saved.")


def load_model_state(model: Transducer, checkpoint: dict):
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        return

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return

    if all(key in checkpoint for key in ("encoder", "decoder", "joint")):
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.decoder.load_state_dict(checkpoint["decoder"])
        model.joint.load_state_dict(checkpoint["joint"])
        return

    raise KeyError(
        "Checkpoint must contain either a full `model` state dict or "
        "`encoder`, `decoder`, and `joint` state dicts."
    )


def initialize_model(
    config: AttrDict, vocab_size: int, device: Union[str, torch.device]
):
    model = Transducer(config.model, vocab_size, device)
    if config.training.load_model:
        checkpoint = torch.load(config.training.load_model, map_location=device)
        load_model_state(model, checkpoint)
    else:
        init_parameters(model, type="uniform")
    model.to(device)
    return model


def adjust_learning_rate(
    optimizer: Optimizer, epoch: int, config: AttrDict, logger: Logger
):
    if config.optim.scheduler == "reduce_on_plateau":
        return

    if (
        epoch >= config.optim.begin_to_adjust_lr
        and epoch % config.optim.adjust_every == 0
    ):
        optimizer.decay_lr()
        if optimizer.lr < 1e-6:
            logger.info("The learning rate is too low to train.")
            return
        logger.info(f"Epoch {epoch} update learning rate: {optimizer.lr:.6f}")

def create_optimizer(model: Transducer, config: AttrDict):
    return Optimizer(model, config)
    
def prepare_data_loaders(config: AttrDict):
    bucket_by_duration = (
        True
        if config.training.bucket_by_duration is None
        else bool(config.training.bucket_by_duration)
    )
    sort_eval_by_duration = (
        True
        if config.training.sort_eval_by_duration is None
        else bool(config.training.sort_eval_by_duration)
    )
    bucket_size_multiplier = config.training.bucket_size_multiplier or 100

    tokenizer = CharTokenizer(
        transcript_path=os.path.join(config.data.name, config.data.core_train), 
        batch_size=config.training.batch_size,
    )

    train_data = build_data_loader(
        os.path.join(config.data.name, config.data.core_train),
        tokenizer,
        batch_size=config.training.batch_size,
        shuffle=True,
        bucket_by_duration=bucket_by_duration and config.training.batch_size > 1,
        bucket_size_multiplier=bucket_size_multiplier,
    )
    test_data = build_data_loader(
        os.path.join(config.data.name, config.data.core_test),
        tokenizer,
        batch_size=config.training.batch_size,
        sort_by_duration=sort_eval_by_duration and config.training.batch_size > 1,
    )
    val_data = build_data_loader(
        os.path.join(config.data.name, config.data.core_val),
        tokenizer,
        batch_size=config.training.batch_size,
        sort_by_duration=sort_eval_by_duration and config.training.batch_size > 1,
    )
    return train_data, test_data, val_data, tokenizer
