import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

def creat_optimizer(model, args):
    return torch.optim.AdamW(model.parameters(), lr=args.lr)


def create_scheduler(optimizer, args):
    if args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.lr_scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.lr_decay_epochs, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_rate, 
                                    patience=args.lr_decay_epochs//2, verbose=True)
    else:
        raise ValueError(f"Unsupported scheduler type: {args.lr_scheduler}")
    
    return scheduler