"""Fine-tune the EfficientNet-B0 detector and write the checkpoint the API loads.

    python ml/scripts/train_cnn.py --corpus datasets --out artifacts/model.pt
    python ml/scripts/train_cnn.py --corpus ml/.smoke/corpus --out ml/.smoke/model.pt \\
        --epochs 3 --limit 240            # pipeline verification, no download

**One logit, ``BCEWithLogitsLoss``.** The prototype ended its head with a softmax over two
classes and then read column 1. That is the same maths with an extra subtraction, but it makes the
checkpoint's output a *pair* that must sum to one, which is what forced the ``fused * 1.6``
stretch later: a softmax probability on an uncalibrated model is compressed toward 0.5, and a
single logit through sigmoid is not. ``app.ml.cnn_runtime`` builds ``Linear(in_features, 1)`` and
refuses anything else, so this script must produce exactly that.

**Class weighting through ``pos_weight``**, not through a resampled loader. Resampling repeats
images within an epoch, which the augmentation makes *less* like the deployment distribution —
the same photograph appears four times with four different JPEG histories and the model learns
that repetition is a class. ``pos_weight`` shifts the loss without touching the data.

**Best-validation checkpoint, then early stopping.** Saving the last epoch rewards the model that
has memorised the most; saving the best-val weights makes the reported number the one the file
actually contains. The checkpoint's metadata records which epoch that was, so
``evaluate_cnn.py`` can be read against it.

**Reproducibility is a claim this script can actually support**: seeds for python, numpy and
torch, a fixed split derived from ``corpus.split_of`` (never from listing order), and the corpus
manifest written beside the weights. Without the manifest, ``metrics.json`` is a number with no
way to re-derive it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import add_paths  # noqa: E402

add_paths()

from truelens_ml.corpus import (  # noqa: E402
    FAKE,
    REAL,
    build_dataset,
    discover,
    split_samples,
    tally,
    write_manifest,
)
from truelens_ml.model import ARCHITECTURE, build_model, save_checkpoint  # noqa: E402


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def average_loss(model, loader, criterion, device) -> tuple[float, list[float], list[int]]:
    """Forward pass only. Returns ``(loss, probabilities, labels)``.

    Probabilities come back as python floats rather than tensors because every caller either
    prints them or hands them to scikit-learn, and a checkpoint of a model whose eval path shares
    a tensor cache with the training loop is how the Grad-CAM ``Inference tensors cannot be saved
    for backward`` error is born.
    """
    import torch

    model.eval()
    total = 0.0
    batches = 0
    probs: list[float] = []
    labels: list[int] = []
    with torch.inference_mode():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logit = model(x).squeeze(-1)
            total += float(criterion(logit, y).item())
            batches += 1
            probs.extend([float(v) for v in torch.sigmoid(logit).detach().cpu().tolist()])
            labels.extend([int(v) for v in y.detach().cpu().tolist()])
    return (total / max(1, batches), probs, labels)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler_scale: float = 1.0) -> float:
    import torch

    model.train()
    total = 0.0
    batches = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logit = model(x).squeeze(-1)
        loss = criterion(logit, y)
        loss.backward()
        # A clipped gradient norm, not a clipped weight: augmentation includes a JPEG round trip,
        # and an image that fails to decode mid-epoch should not be able to move the model.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.item())
        batches += 1
    return total / max(1, batches)


def choose_threshold(labels, probs) -> float:
    """The F1-maximising operating point on the validation set.

    Reported alongside every threshold-dependent metric. 0.5 is the default of a classifier that
    was never asked what its output means; on this task a false positive is a photograph wrongly
    accused and a false negative is a forgery waved through, and the product page says the
    difference out loud, so the number has to be chosen rather than assumed.
    """
    import numpy as np
    from sklearn.metrics import f1_score

    y = np.asarray(labels)
    p = np.asarray(probs)
    candidates = np.unique(np.clip(p, 0.0, 1.0))
    if candidates.size == 0:
        return 0.5
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def main() -> int:  # noqa: C901 - a training loop reads top to bottom or not at all
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="", help="checkpoint path (default: artifacts/model.pt)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers; 0 on Windows, where spawn re-imports torch per child")
    ap.add_argument("--patience", type=int, default=4, help="val-AUC epochs without improvement")
    ap.add_argument("--fractions", default="0.7,0.15,0.15")
    ap.add_argument("--limit", type=int, default=None, help="cap corpus size, balanced by class")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--log", default="", help="write a per-epoch JSON log here")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from truelens_ml.augment import IMAGENET_MEAN, IMAGENET_STD

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fractions = [float(f) for f in args.fractions.split(",")]

    samples = discover(args.corpus, limit=args.limit)
    splits = split_samples(samples, fractions=fractions, seed=args.seed)
    print(f"corpus {args.corpus}: {tally(samples)}")
    for name, part in splits.items():
        summary = tally(part)
        print(f"  {name:5s} {summary['images']:6d} images "
              f"(real {summary['real']}, fake {summary['fake']}, {summary['groups']} groups)")

    train_ds = build_dataset(splits["train"], img_size=args.img_size, augment=True,
                             mean=IMAGENET_MEAN, std=IMAGENET_STD)
    val_ds = build_dataset(splits["val"], img_size=args.img_size, augment=False,
                           mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=False,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers)

    model = build_model(pretrained=not args.no_pretrained)
    init = getattr(model, "truelens_init", "unknown")
    model.to(device)

    n_real = sum(1 for s in splits["train"] if s.label == REAL)
    n_fake = sum(1 for s in splits["train"] if s.label == FAKE)
    # pos_weight scales the positive term so equal errors on both classes cost the same.
    pos_weight = torch.tensor([n_real / max(1, n_fake)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    out = args.out or str(Path("artifacts") / "model.pt")
    best_auc = -1.0
    best_epoch = -1
    best_path = os.path.join(os.path.dirname(os.path.abspath(out)) or ".", "best_model.pt")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    history: list[dict[str, object]] = []
    stale = 0
    t0 = time.time()

    print(f"\ninit={init}  device={device}  pos_weight={float(pos_weight):.3f}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_probs, val_labels = average_loss(model, val_loader, criterion, device)
        auc = _auc(val_labels, val_probs)
        threshold = choose_threshold(val_labels, val_probs)
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_auc={auc:.4f} thr={threshold:.3f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s")
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "val_loss": round(val_loss, 6), "val_auc": round(auc, 6),
                        "val_threshold": round(threshold, 4),
                        "lr": optimizer.param_groups[0]["lr"]})

        if auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = auc, epoch, 0
            meta = {"architecture": ARCHITECTURE, "epoch": epoch, "val_auc": round(auc, 6),
                    "val_threshold": round(threshold, 4), "init": init,
                    "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "img_size": args.img_size, "seed": args.seed,
                    "corpus": str(Path(args.corpus).resolve()),
                    "splits": {k: tally(v)["images"] for k, v in splits.items()},
                    "pos_weight": round(float(pos_weight), 6)}
            save_checkpoint(model, best_path, meta=meta)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}: no val-AUC improvement for "
                      f"{args.patience} epochs (best was epoch {best_epoch}, {best_auc:.4f})")
                break

    if best_epoch < 0:
        print("no epoch improved validation AUC; refusing to write a checkpoint of the last "
              "epoch, which would be the most over-fitted set of weights in the run.")
        return 1

    final = torch.load(best_path, map_location="cpu", weights_only=True)
    save_checkpoint_from_state(model, final["state_dict"], out, final.get("meta") or {})
    os.remove(best_path)

    manifest_path = os.path.splitext(out)[0] + ".manifest.json"
    write_manifest(manifest_path, splits)
    if args.log:
        Path(args.log).write_text(json.dumps({
            "config": vars(args), "history": history,
            "best_epoch": best_epoch, "best_val_auc": round(best_auc, 6),
            "corpus": tally(samples), "seconds": round(time.time() - t0, 1),
        }, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nwrote {out} (epoch {best_epoch}, val AUC {best_auc:.4f})")
    print(f"wrote {manifest_path}")
    return 0


def save_checkpoint_from_state(model, state, path, meta) -> None:
    """Persist the best weights without re-running training to reproduce them.

    The in-memory model is whatever the last epoch reached by the time the loop stops; the best
    weights are on disk. Loading them back and saving from there is the only way the written file
    is guaranteed to be the file the validation number describes.
    """
    import torch

    model.load_state_dict(state)
    save_checkpoint(model, path, meta=meta)


def _auc(labels, probs) -> float:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels)
    if len(np.unique(y)) < 2:
        return 0.0
    return float(roc_auc_score(y, np.asarray(probs)))


if __name__ == "__main__":
    raise SystemExit(main())
