from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PAPER_IPIX_LABELS = [
    ("1", "19931107_135603_starea"),
    ("2", "19931107_141630_starea"),
    ("3", "19931107_145028_starea"),
    ("4", "19931108_213827_starea"),
    ("5", "19931108_220902_starea"),
    ("6", "19931109_191449_starea"),
    ("7", "19931109_202217_starea"),
    ("8", "19931110_001635_starea"),
    ("9", "19931111_163625_starea"),
    ("10", "19931118_023604_stareC0000"),
    ("11", "19931118_035737_stareC0000"),
    ("12", "19931118_162155_stareC0000"),
    ("13", "19931118_162658_stareC0000"),
    ("14", "19931118_174259_stareC0000"),
]


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("读取 checkpoint 元信息需要安装 torch。") from exc

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {}
    return {"epoch": payload.get("epoch"), "best_loss": payload.get("best_loss"), "args": payload.get("args", {})}


def overall_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pfa, payload in sorted(results["pfa"].items(), key=lambda item: float(item[0])):
        rows.append(
            {
                "Pfa": pfa,
                "Threshold": payload["threshold"],
                "PD": payload["PD"],
                "PF": payload["PF"],
                "TP": payload["TP"],
                "FN": payload["FN"],
                "FP": payload["FP"],
                "TN": payload["TN"],
            }
        )
    return rows


def grouped_by_polarization(results: dict[str, Any], pfa: str) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, int]] = {}
    for row in results["pfa"][pfa]["per_file"]:
        pol = row["polarization"]
        groups.setdefault(pol, {"TP": 0, "FN": 0, "FP": 0, "TN": 0})
        for key in ["TP", "FN", "FP", "TN"]:
            groups[pol][key] += int(row[key])

    rows = []
    for pol in sorted(groups):
        counts = groups[pol]
        tp, fn, fp, tn = counts["TP"], counts["FN"], counts["FP"], counts["TN"]
        rows.append(
            {
                "Polarization": pol,
                "PD": tp / (tp + fn) if tp + fn else 0.0,
                "PF": fp / (fp + tn) if fp + tn else 0.0,
                **counts,
            }
        )
    return rows


def worst_file_rows(results: dict[str, Any], pfa: str, limit: int = 12) -> list[dict[str, Any]]:
    rows = sorted(results["pfa"][pfa]["per_file"], key=lambda row: float(row["PD"]))[:limit]
    return [
        {
            "Source": row["source"],
            "Polarization": row["polarization"],
            "PD": row["PD"],
            "PF": row["PF"],
            "TP": row["TP"],
            "FN": row["FN"],
            "FP": row["FP"],
            "TN": row["TN"],
        }
        for row in rows
    ]


def paper_label_rows(results: dict[str, Any], pfa: str) -> list[dict[str, Any]]:
    per_file = results["pfa"][pfa]["per_file"]
    pd_by_key = {(row["source"], row["polarization"]): float(row["PD"]) for row in per_file}
    rows = []
    for label, source in PAPER_IPIX_LABELS:
        rows.append(
            {
                "Label": label,
                "Source": source,
                "HH": pd_by_key[(source, "hh")],
                "HV": pd_by_key[(source, "hv")],
                "VV": pd_by_key[(source, "vv")],
                "VH": pd_by_key[(source, "vh")],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("生成实验图片需要安装 matplotlib：pip install matplotlib") from exc
    return plt


def plot_overall(path: Path, rows: list[dict[str, Any]]) -> None:
    plt = require_matplotlib()
    pfa = [float(row["Pfa"]) for row in rows]
    pd = [float(row["PD"]) for row in rows]
    pf = [float(row["PF"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.plot(pfa, pd, marker="o", linewidth=2, label="PD")
    ax.plot(pfa, pf, marker="s", linewidth=2, label="PF")
    ax.set_xscale("log")
    ax.set_xlabel("Pfa")
    ax.set_ylabel("Rate")
    ax.set_title("IPIX evaluation across Pfa")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_polarization(path: Path, rows: list[dict[str, Any]]) -> None:
    plt = require_matplotlib()
    labels = [row["Polarization"] for row in rows]
    pd = [float(row["PD"]) for row in rows]
    pf = [float(row["PF"]) for row in rows]
    x = list(range(len(labels)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.bar([i - width / 2 for i in x], pd, width=width, label="PD")
    ax.bar([i + width / 2 for i in x], pf, width=width, label="PF")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Polarization")
    ax.set_ylabel("Rate")
    ax.set_title("Pfa=0.001 by polarization")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_worst_files(path: Path, rows: list[dict[str, Any]]) -> None:
    plt = require_matplotlib()
    labels = [f"{row['Source']} / {row['Polarization']}" for row in rows]
    pd = [float(row["PD"]) for row in rows]

    fig_height = max(5.2, 0.38 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(9.5, fig_height), constrained_layout=True)
    y = list(range(len(labels)))
    ax.barh(y, pd, color="#4C78A8")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("PD")
    ax.set_title("Lowest PD file/polarization pairs at Pfa=0.001")
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion(path: Path, row: dict[str, Any]) -> None:
    plt = require_matplotlib()
    labels = ["TP", "FN", "FP", "TN"]
    values = [int(row[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#59A14F", "#E15759", "#F28E2B", "#76B7B2"])
    ax.set_ylabel("Bins")
    ax.set_title("Confusion counts at Pfa=0.001")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_paper_label_style(path: Path, rows: list[dict[str, Any]]) -> None:
    plt = require_matplotlib()
    labels = [int(row["Label"]) for row in rows]
    series = [("HH", "hh", "o"), ("HV", "hv", "s"), ("VV", "vv", "^"), ("VH", "vh", "D")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, (column, title, marker) in zip(axes.ravel(), series):
        ax.plot(labels, [float(row[column]) for row in rows], color="#D62728", marker=marker, linewidth=1.8)
        ax.set_title(title.upper())
        ax.set_xticks(labels)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Paper data label")
        ax.set_ylabel("Detection probability")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Reproduced ST-GNN result aligned with paper Fig. 7 (Pfa=0.001)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_float(value: Any, digits: int = 10) -> str:
    return f"{float(value):.{digits}g}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(out)


def write_report(
    path: Path,
    results: dict[str, Any],
    overall: list[dict[str, Any]],
    by_pol: list[dict[str, Any]],
    worst: list[dict[str, Any]],
    paper_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    args = metadata.get("args", {})
    pols = args.get("pols", ["hh", "hv", "vv", "vh"])
    if not isinstance(pols, list):
        pols = ["hh", "hv", "vv", "vh"]
    pfa_001 = next(row for row in overall if row["Pfa"] == "0.001")
    best_epoch = metadata.get("epoch", "未知")
    best_loss = metadata.get("best_loss", "未知")

    text = f"""# IPIX 实验日志提取与论文对比

本报告由本地实验产物提取生成，结果文件以 `checkpoints/ipix_eval_results.json` 为准。当前工作区没有发现单独保存的训练 stdout `.log` 文件；原始脚本只在终端打印 epoch 进度，因此本次已完成实验无法还原真实逐 epoch 训练曲线。`train_ipix.py` 已补充训练历史记录功能，后续重新训练会自动生成 `ipix_training_history.csv` 和 `ipix_training_curves.png`。

## 实验产物来源

- 评估结果：`checkpoints/ipix_eval_results.json`
- 最优模型：`checkpoints/ipix_best_model.pth`
- 最终模型：`checkpoints/ipix_final_model.pth`
- 配置文件：`configs/ipix_stgnn.yaml`

## 运行配置

- 数据集：IPIX Dartmouth 预处理窗口
- 数据目录：`{args.get("data_dir", "datasets/ipix_dartmouth/processed/window4_stride4_related")}`
- 极化方式：`{", ".join(pols)}`
- 模型输入：`P={args.get("P", 4)}`，`N={args.get("N", 14)}`
- 训练轮数：`{args.get("epochs", 50)}`
- 批大小：`{args.get("batch_size", 32)}`
- 学习率：`{args.get("lr", 0.001)}`
- 随机种子：`{args.get("seed", 42)}`
- 最优 checkpoint epoch：`{best_epoch}`
- 最优 loss：`{best_loss}`
- 测试文件数：`{results["num_files"]}`
- 用于阈值估计的杂波单元数：`{results["num_clutter_bins"]:,}`

## 总体评估结果

在目标 `Pfa=0.001` 下，整体 `PD={format_float(pfa_001["PD"])}`，实际 `PF={format_float(pfa_001["PF"])}`。随着 Pfa 放宽，PD 从 `0.8474` 提升到 `0.9973`，但误警率也同步升高。

![不同 Pfa 下的 PD/PF 曲线](figures/ipix_pfa_pd_pf.png)

{md_table(
        ["Pfa", "Threshold", "PD", "PF", "TP", "FN", "FP", "TN"],
        [
            [
                row["Pfa"],
                format_float(row["Threshold"]),
                format_float(row["PD"]),
                format_float(row["PF"]),
                f"{int(row['TP']):,}",
                f"{int(row['FN']):,}",
                f"{int(row['FP']):,}",
                f"{int(row['TN']):,}",
            ]
            for row in overall
        ],
    )}

## 与原论文 Fig. 7 的对比

论文 Fig. 7 在 `Pfa=0.001` 下比较了 IPIX 14 个数据集、4 种极化方式上的检测概率，并同时给出了 MDCCNN、DPA、SVM、Tri-feature 和 ST-GNN 五类方法。下面的图将本次复现实验结果按论文 Table I 的 label 顺序重新排列，便于直接对照论文中的红色 ST-GNN 曲线。

![按论文 Fig. 7 标签顺序排列的 ST-GNN 复现结果](figures/ipix_paper_fig7_style_pfa_0.001.png)

对比结论：

- 趋势基本一致：label 6、label 7 是主要困难场景，检测概率明显低于其他数据集。
- 极化差异一致：`hv` 和 `vh` 整体更稳定，`vv` 相对更弱，`hh` 在困难场景下降更明显。
- 整体水平接近论文 ST-GNN 曲线的高检测区间，多数 label 的 PD 在 `0.95` 以上。
- 该结果更适合作为“复现趋势和量级一致”的证据；当前阈值由评估结果中的杂波分布计算，且 `target_policy=related` 的标签口径可能与论文实现存在细节差异，因此不应直接表述为严格超过原论文。

{md_table(
        ["Label", "数据源", "HH", "HV", "VV", "VH"],
        [
            [
                row["Label"],
                row["Source"],
                format_float(row["HH"], 4),
                format_float(row["HV"], 4),
                format_float(row["VV"], 4),
                format_float(row["VH"], 4),
            ]
            for row in paper_rows
        ],
    )}

## Pfa=0.001 的极化对比

`hv` 与 `vh` 的检测概率最高且误警率较低；`vv` 的 PD 最低、PF 最高，是当前模型在该批 IPIX 结果中的主要短板。

![Pfa=0.001 下各极化 PD/PF 对比](figures/ipix_polarization_pfa_0.001.png)

{md_table(
        ["极化", "PD", "PF", "TP", "FN", "FP", "TN"],
        [
            [
                row["Polarization"],
                format_float(row["PD"]),
                format_float(row["PF"]),
                f"{int(row['TP']):,}",
                f"{int(row['FN']):,}",
                f"{int(row['FP']):,}",
                f"{int(row['TN']):,}",
            ]
            for row in by_pol
        ],
    )}

## Pfa=0.001 的低 PD 样本

低 PD 样本集中在 `19931109_191449_starea`、`19931109_202217_starea` 以及部分 `vv` 极化文件，说明不同场景和极化之间存在明显难度差异。

![Pfa=0.001 下最低 PD 的文件/极化组合](figures/ipix_worst_files_pfa_0.001.png)

{md_table(
        ["数据源", "极化", "PD", "PF", "TP", "FN", "FP", "TN"],
        [
            [
                row["Source"],
                row["Polarization"],
                format_float(row["PD"]),
                format_float(row["PF"]),
                f"{int(row['TP']):,}",
                f"{int(row['FN']):,}",
                f"{int(row['FP']):,}",
                f"{int(row['TN']):,}",
            ]
            for row in worst
        ],
    )}

## 混淆计数概览

![Pfa=0.001 下 TP/FN/FP/TN 计数](figures/ipix_confusion_pfa_0.001.png)

## 后续复现建议

为了让结果更接近论文口径，下一步建议把 FAR 阈值改为由训练集杂波样本估计，再生成同样的 Fig. 7 风格曲线。这样可以避免测试集阈值带来的乐观偏差，也更适合写成正式论文复现对比。

## 导出的辅助文件

- `reports/ipix_eval_overall.csv`
- `reports/ipix_eval_pfa_0.001_by_polarization.csv`
- `reports/ipix_eval_pfa_0.001_worst_files.csv`
- `reports/ipix_eval_paper_fig7_style_pfa_0.001.csv`
- `reports/figures/ipix_pfa_pd_pf.png`
- `reports/figures/ipix_paper_fig7_style_pfa_0.001.png`
- `reports/figures/ipix_polarization_pfa_0.001.png`
- `reports/figures/ipix_worst_files_pfa_0.001.png`
- `reports/figures/ipix_confusion_pfa_0.001.png`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Chinese IPIX experiment report and figures.")
    parser.add_argument("--results", type=Path, default=Path("checkpoints/ipix_eval_results.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ipix_best_model.pth"))
    parser.add_argument("--report", type=Path, default=Path("reports/ipix_experiment_log.md"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--target-pfa", default="0.001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.results)
    metadata = load_checkpoint_metadata(args.checkpoint)
    overall = overall_rows(results)
    by_pol = grouped_by_polarization(results, args.target_pfa)
    worst = worst_file_rows(results, args.target_pfa)
    paper_rows = paper_label_rows(results, args.target_pfa)

    write_csv(Path("reports/ipix_eval_overall.csv"), overall)
    write_csv(Path("reports/ipix_eval_pfa_0.001_by_polarization.csv"), by_pol)
    write_csv(Path("reports/ipix_eval_pfa_0.001_worst_files.csv"), worst)
    write_csv(Path("reports/ipix_eval_paper_fig7_style_pfa_0.001.csv"), paper_rows)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_overall(args.figures_dir / "ipix_pfa_pd_pf.png", overall)
    plot_polarization(args.figures_dir / "ipix_polarization_pfa_0.001.png", by_pol)
    plot_worst_files(args.figures_dir / "ipix_worst_files_pfa_0.001.png", worst)
    plot_confusion(args.figures_dir / "ipix_confusion_pfa_0.001.png", next(row for row in overall if row["Pfa"] == args.target_pfa))
    plot_paper_label_style(args.figures_dir / "ipix_paper_fig7_style_pfa_0.001.png", paper_rows)
    write_report(args.report, results, overall, by_pol, worst, paper_rows, metadata)


if __name__ == "__main__":
    main()
