

def run_experiment(config: Dict, device: torch.device) -> TrainResult:
    name = config.get("exp_name", "experiment")
    out_dir = config.get("out_dir", "results")
    ensure_dir(out_dir)
    ensure_dir("models")

    root = config["root"]
    view_names: List[str] = config.get("views") or []
    if not view_names:
        raise ValueError("config['views'] must be a non-empty list of view folder names")
    modes = config.get("modes") or infer_modes(view_names)

    tfms = build_transforms(modes, image_size=config.get("image_size", 224), augment=True)

    dataset_type = config.get("datasetType", "multiview")
    ds = MultiViewImageFolder(root=root, view_names=view_names, modes=modes, transform=tfms, dataset_type=dataset_type)
    train_ds, val_ds = ds.split(train_ratio=float(config.get("train_ratio", 0.85)))

    use_sampler = bool(config.get("use_sampler", False))
    if use_sampler:
        from .utils import make_balanced_sampler

        labels = [s[-1] for s in train_ds.samples]
        sampler = make_balanced_sampler(labels)
        loaders = {
            "train": DataLoader(train_ds, batch_size=int(config.get("batch_size", 8)), sampler=sampler),
            "val": DataLoader(val_ds, batch_size=int(config.get("batch_size", 8)), shuffle=False),
        }
    else:
        loaders = {
            "train": DataLoader(train_ds, batch_size=int(config.get("batch_size", 8)), shuffle=True),
            "val": DataLoader(val_ds, batch_size=int(config.get("batch_size", 8)), shuffle=False),
        }
    sizes = {k: len(v.dataset) for k, v in loaders.items()}

    # channels per view derived from modes
    channels = [3 if m == "RGB" else 1 for m in modes]
    num_classes = len(train_ds.classes)

    which = str(config.get("backbone", "resnet18")).lower()
    variant = str(config.get("variant", ("concat" if dataset_type == "concat" else "multiview"))).lower()

    if variant == "concat":
        in_ch = sum(channels)
        if which == "resnet18":
            model = ConcatResNet18(in_ch, num_classes, pretrained=bool(config.get("pretrained", True)))
        elif which == "resnet34":
            model = ConcatResNet34(in_ch, num_classes, pretrained=bool(config.get("pretrained", True)))
        else:
            model = ConcatResNet50(in_ch, num_classes, pretrained=bool(config.get("pretrained", True)))
    else:
        if which == "resnet18":
            model_cls = MultiViewResNet18
            feat_gated = 512
        elif which == "resnet34":
            model_cls = MultiViewResNet34
            feat_gated = 512
        else:
            model_cls = MultiViewResNet50
            feat_gated = 2048
        model = model_cls(
            channels=channels,
            num_classes=num_classes,
            gated=bool(config.get("gated", True)),
            pretrained=bool(config.get("pretrained", True)),
        )
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config.get("lr", 0.001)), momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.get("lr_step", 7)), gamma=float(config.get("lr_gamma", 0.1)))

    model, tr_l, va_l, tr_a, va_a, tr_f1, va_f1, gates = train_model(
        model,
        loaders,
        sizes,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=int(config.get("epochs", 10)),
        model_name=name,
    )

    model_path = os.path.join("models", f"{name}.pt")
    torch.save(model.state_dict(), model_path)

    # Optionally persist gates summary
    gates_path = None
    if gates:
        gates_path = os.path.join(out_dir, f"gates_{name}.pt")
        torch.save(gates, gates_path)

    return TrainResult(
        model_path=model_path,
        train_losses=tr_l,
        val_losses=va_l,
        train_accs=tr_a,
        val_accs=va_a,
        train_f1s=tr_f1,
        val_f1s=va_f1,
        val_gates=gates,
    )