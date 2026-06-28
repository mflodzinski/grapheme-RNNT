import inspect

from torch import optim


def config_value(config, name, default=None):
    value = getattr(config, name)
    return default if value is None else value


def parse_betas(value):
    if value is None:
        return (0.9, 0.999)
    if isinstance(value, (list, tuple)):
        return tuple(map(float, value))
    return tuple(map(float, value.strip("()").split(",")))


class Optimizer(object):
    def __init__(self, model, config):
        self.config = config
        self.optimizer = build_optimizer(model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        self.global_step = 1
        self.current_epoch = 0
        self.lr = self.optimizer.param_groups[0]["lr"]
        self.decay_ratio = config_value(config, "decay_ratio", 1.0)
        self.epoch_decay_flag = False

    def step(self):
        self.global_step += 1
        self.optimizer.step()

    def epoch(self):
        self.current_epoch += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def decay_lr(self):
        self.lr = max(self.decay_ratio * self.lr, self.config.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def step_scheduler(self, metric, logger=None):
        if self.scheduler is None:
            return

        old_lr = self.lr
        self.scheduler.step(metric)
        self.lr = self.optimizer.param_groups[0]["lr"]

        if logger is not None and self.lr != old_lr:
            logger.info(
                f"ReduceLROnPlateau updated learning rate: {old_lr:.6f} -> {self.lr:.6f}"
            )


def get_optim_groups(model, weight_decay):
    parameters = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in parameters if p.dim() >= 2]
    nondecay_params = [p for p in parameters if p.dim() < 2]
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nondecay_params, "weight_decay": 0.0},
    ]


def build_optimizer(model, config):
    weight_decay = config_value(config, "weight_decay", 0.0)
    params = get_optim_groups(model, weight_decay)
    if config.type == "adamw":
        kwargs = {
            "params": params,
            "weight_decay": weight_decay,
            "lr": config.lr,
            "betas": parse_betas(config_value(config, "betas", None)),
            "eps": float(config_value(config, "eps", 1e-8)),
        }
        if (
            config_value(config, "fused", None) is not None
            and "fused" in inspect.signature(optim.AdamW).parameters
        ):
            kwargs["fused"] = bool(config.fused)
        return optim.AdamW(**kwargs)
    elif config.type == "sgd":
        return optim.SGD(
            params=params,
            weight_decay=weight_decay,
            lr=config.lr,
            momentum=config_value(config, "momentum", 0.0),
            nesterov=bool(config_value(config, "nesterov", False)),
        )
    else:
        raise NotImplementedError


def build_scheduler(optimizer, config):
    if config_value(config, "scheduler", "step_decay") != "reduce_on_plateau":
        return None

    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config_value(config, "plateau_factor", 0.5)),
        patience=int(config_value(config, "plateau_patience", 2)),
        threshold=float(config_value(config, "plateau_threshold", 1e-4)),
        min_lr=float(config_value(config, "min_lr", 1e-6)),
    )
