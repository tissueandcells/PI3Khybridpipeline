"""
train_mtgnn.py — Train MT-GNN v1 or v2 on curated PI3K dataset.

Usage:
    python train_mtgnn.py --config configs/v1.yaml
    python train_mtgnn.py --config configs/v2.yaml

v1: baseline (22-d features, plain MSE)
v2: selectivity-aware (32-d features, inverse-frequency weights,
    pairwise ranking loss weight 0.3)

Produces:
    models/mtgnn_{v1,v2}.pt — model weights
    data/curated/test_predictions_{v1,v2}.csv — test-set predictions
    logs/train_{v1,v2}.json — training curves
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Actual Dataset/DataLoader classes implemented in dataset.py (see below)
# from dataset import PI3KDataset, collate_fn
# from model import MTGNN, masked_mse_loss, pairwise_ranking_loss

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    print(f"→ Training MT-GNN with config: {args.config}")
    print(json.dumps(cfg, indent=2))

    # --- Pseudocode skeleton; full implementation in final repo ---
    #
    # train_ds = PI3KDataset(split="train", feat_dim=cfg["in_dim"])
    # val_ds   = PI3KDataset(split="val",   feat_dim=cfg["in_dim"])
    # test_ds  = PI3KDataset(split="test",  feat_dim=cfg["in_dim"])
    #
    # model = MTGNN(in_dim=cfg["in_dim"], hidden_dim=256).to(DEVICE)
    # opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    #
    # best_val = float("inf")
    # patience_counter = 0
    # for epoch in range(cfg["epochs"]):
    #     train_loss = train_one_epoch(model, train_loader, opt, cfg)
    #     val_loss   = evaluate(model, val_loader)
    #     if val_loss < best_val:
    #         best_val = val_loss
    #         torch.save(model.state_dict(), ROOT / "models" / cfg["checkpoint_name"])
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1
    #     if patience_counter >= cfg["patience"]:
    #         print(f"Early stopping at epoch {epoch}")
    #         break
    #
    # # Final evaluation on test set
    # model.load_state_dict(torch.load(ROOT / "models" / cfg["checkpoint_name"]))
    # test_preds = predict(model, test_loader)
    # test_preds.to_csv(
    #     ROOT / "data" / "curated" / f"test_predictions_{cfg['variant']}.csv",
    #     index=False,
    # )

    print("→ See notebooks/train_mtgnn_example.ipynb for full runnable notebook.")


if __name__ == "__main__":
    main()
