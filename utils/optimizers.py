import torch.optim
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model(model):
    for name, moudule in model.named_modules():
        moudule.eval()
        for param in moudule.parameters():
            param.requires_grad = False


def get_optmizer_params_num(optimizer):
    tot_params = 0
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if param.requires_grad:
                tot_params += param.numel()
    return tot_params


def creat_optimizer(model, args, **kwargs):
    params = list()

    adapter = kwargs.get("adapter", None)
    if adapter is not None:
        # Adapter
        params += list(adapter.parameters())
    else:
        # Full finetune
        params += model.parameters()

    optim = torch.optim.AdamW(params, lr=args.lr)
    logger.info(
        f"Total number of trainable parameters: {get_optmizer_params_num(optim) / 1000000:.2f}M"
    )
    return optim


def create_scheduler(optimizer, args):
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.lr_scheduler == "step":
        scheduler = StepLR(
            optimizer, step_size=args.lr_decay_epochs, gamma=args.lr_decay_rate
        )
    elif args.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_decay_rate,
            patience=args.lr_decay_epochs // 2,
            verbose=True,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {args.lr_scheduler}")

    return scheduler
