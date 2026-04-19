"""
evaluate_controls.py — Evaluate MT-GNN v1/v2 on 9 positive controls + 22 decoys.

Produces:
    data/controls/predictions_v1_v2.csv — per-compound predictions
    data/controls/metrics_summary.json  — layer-stratified AUROC, specificity, etc.

Reported in manuscript Sections 3.5 and 3.9, Tables 5-9, Figure 5.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# from model import MTGNN
# from features import mol_to_graph_features

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "controls"


def load_controls() -> pd.DataFrame:
    pos = pd.read_csv(DATA_DIR / "positive_controls.csv")
    neg = pd.read_csv(DATA_DIR / "negative_controls.csv")
    pos["cohort"] = "positive"
    neg["cohort"] = neg["layer"].map({"A": "layerA", "B": "layerB", "C": "layerC"})
    return pd.concat([pos, neg], ignore_index=True)


def predict_with_model(checkpoint: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Run MT-GNN inference on control compounds. Stub — implement with actual model."""
    # model = MTGNN(in_dim=32 if "v2" in checkpoint.name else 22)
    # model.load_state_dict(torch.load(checkpoint))
    # model.eval()
    # preds = []
    # for smi in df.smiles:
    #     graph = mol_to_graph_features(smi)
    #     with torch.no_grad():
    #         pred = model(graph.x, graph.edge_index, graph.batch).cpu().numpy()
    #     preds.append(pred.flatten())
    # return pd.DataFrame(preds, columns=["pic50_alpha", "pic50_beta", "pic50_delta", "pic50_gamma"])
    raise NotImplementedError("Load actual trained weights from models/")


def compute_metrics(df: pd.DataFrame) -> dict:
    # Build binary labels for classification
    df = df.copy()
    df["label"] = (df["cohort"] == "positive").astype(int)
    df["max_pred_pic50"] = df[
        ["pred_alpha", "pred_beta", "pred_delta", "pred_gamma"]
    ].max(axis=1)

    pos_mask = df["cohort"] == "positive"
    metrics = {}
    for layer in ["layerA", "layerB", "layerC"]:
        layer_mask = (df["cohort"] == layer) | pos_mask
        sub = df[layer_mask]
        if sub["label"].nunique() >= 2:
            metrics[f"auroc_{layer}"] = roc_auc_score(sub["label"], sub["max_pred_pic50"])
            metrics[f"aupr_{layer}"] = average_precision_score(sub["label"], sub["max_pred_pic50"])
        else:
            metrics[f"auroc_{layer}"] = None

    # Overall
    metrics["auroc_overall"] = roc_auc_score(df["label"], df["max_pred_pic50"])
    metrics["aupr_overall"] = average_precision_score(df["label"], df["max_pred_pic50"])

    # Filter specificity at pIC50 > 7 threshold
    decoys = df[~pos_mask]
    fp_rate = (decoys["max_pred_pic50"] > 7).mean()
    metrics["filter_specificity"] = 1.0 - fp_rate

    return metrics


def main() -> None:
    df = load_controls()
    print(f"Loaded {len(df)} control compounds ({(df.cohort=='positive').sum()} positive, "
          f"{(df.cohort != 'positive').sum()} negative)")

    # Run v1 and v2 inference
    # df_v1 = predict_with_model(ROOT / "models" / "mtgnn_v1.pt", df)
    # df_v2 = predict_with_model(ROOT / "models" / "mtgnn_v2.pt", df)
    #
    # df = df.join(df_v1.add_prefix("v1_"))
    # df = df.join(df_v2.add_prefix("v2_"))
    # df.to_csv(DATA_DIR / "predictions_v1_v2.csv", index=False)
    #
    # metrics_v1 = compute_metrics(df.rename(columns=lambda c: c.replace("v1_", "pred_")))
    # metrics_v2 = compute_metrics(df.rename(columns=lambda c: c.replace("v2_", "pred_")))
    # (DATA_DIR / "metrics_summary.json").write_text(
    #     json.dumps({"v1": metrics_v1, "v2": metrics_v2}, indent=2)
    # )
    print("→ See notebooks/dual_controls.ipynb for full runnable notebook.")


if __name__ == "__main__":
    main()
