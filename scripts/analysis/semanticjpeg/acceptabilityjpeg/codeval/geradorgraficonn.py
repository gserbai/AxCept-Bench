import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Edit config.py to set your paths before running this script
from config import CSV_OUTPUT, CHARTS_OUTPUT_DIR

# ── Output folder ─────────────────────────────────────────────────────────────
os.makedirs(CHARTS_OUTPUT_DIR, exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_OUTPUT)

# FIX 1: column names match the English headers written by val.py
df = df.rename(columns={
    "Model":            "Model",
    "Test_Dataset":     "Test_Dataset",
    "Overall_Accuracy": "Overall_Accuracy",
})

# FIX 2: enforce consistent label order across all charts
# (non-approx = no error, 1e-1 = highest error rate)
RATE_ORDER = ["non-approx", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]

model_order   = [r for r in RATE_ORDER if r in df["Model"].unique()]
dataset_order = [r for r in RATE_ORDER if r in df["Test_Dataset"].unique()]

df["Model"]        = pd.Categorical(df["Model"],        categories=model_order,   ordered=True)
df["Test_Dataset"] = pd.Categorical(df["Test_Dataset"], categories=dataset_order, ordered=True)
df = df.sort_values(["Model", "Test_Dataset"])

# ── Shared style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Use PDF-safe fonts (no Type 3 fonts — required by most journals)
plt.rcParams["pdf.fonttype"] = 42   # TrueType fonts embedded in PDF
plt.rcParams["ps.fonttype"]  = 42

# Semantic colour palette: green (safe) → red (critical)
cores_gradiente  = sns.color_palette("RdYlGn_r", len(RATE_ORDER))
paleta_semantica = {label: cores_gradiente[i] for i, label in enumerate(RATE_ORDER)}


def save(fig, name):
    """Save figure as both PDF (vector, for paper) and PNG (raster, for preview)."""
    pdf_path = os.path.join(CHARTS_OUTPUT_DIR, f"{name}.pdf")
    png_path = os.path.join(CHARTS_OUTPUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  Saved PDF : {pdf_path}")
    print(f"  Saved PNG : {png_path}")


# =============================================================================
# CHART 1 — Grouped bar chart: Overall Accuracy
# =============================================================================
print("\nGenerating Chart 1: Grouped Bar Chart (Overall Accuracy)...")

fig, ax = plt.subplots(figsize=(14, 7))

sns.barplot(
    data=df,
    x="Model",
    y="Overall_Accuracy",
    hue="Test_Dataset",
    hue_order=dataset_order,
    palette=paleta_semantica,
    edgecolor="black",
    linewidth=0.7,
    ax=ax,
)

# Annotate each bar with its accuracy value
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f",
        label_type="edge",
        fontsize=7,
        padding=2,
        rotation=90,
    )

# Baseline reference line — non-approx model on non-approx dataset
baseline_rows = df[(df["Model"] == "non-approx") & (df["Test_Dataset"] == "non-approx")]
if not baseline_rows.empty:
    baseline = baseline_rows["Overall_Accuracy"].values[0]
    ax.axhline(
        baseline,
        color="black",
        linewidth=1.2,
        linestyle="--",
        alpha=0.3,
        label=f"Baseline (non-approx / non-approx): {baseline:.1f}%",
    )


ax.set_xlabel("Training Context (Model)", fontweight="bold")
ax.set_ylabel("Validation Accuracy (%)", fontweight="bold")

import matplotlib.patches as mpatches

min_acc = df["Overall_Accuracy"].min()
ax.set_ylim(max(0, min_acc - 5), 102)
ax.set_yticks([t for t in ax.get_yticks() if t <= 100])

label_names = {
    "non-approx": "Non-approx",
    "1e-5":       "1e-5",
    "1e-4":       "1e-4",
    "1e-3":       "1e-3",
    "1e-2":       "1e-2",
    "1e-1":       "1e-1",
}
patches = [
    mpatches.Patch(facecolor=paleta_semantica[r], edgecolor="black", linewidth=0.6, label=label_names.get(r, r))
    for r in dataset_order
]
if not baseline_rows.empty:
    patches.append(plt.Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label=f"Baseline (non-approx): {baseline:.1f}%"))

leg = ax.legend(
    handles=patches,
    title="Validated On (Test Scenario):",
    title_fontsize=10,
    fontsize=9,
    loc="lower left",
    bbox_to_anchor=(0, 1.02),
    borderaxespad=0,
    ncol=len(patches),
    frameon=True,
    framealpha=0.95,
    edgecolor="#bbbbbb",
    borderpad=0.9,
    labelspacing=0.8,
    handlelength=1.4,
    handleheight=1.1,
)
leg.get_title().set_fontweight("bold")

plt.tight_layout()
fig.subplots_adjust(top=0.82)
save(fig, "01_Overall_Accuracy_BarChart")
plt.close()


# =============================================================================
# CHART 2 — Per-model heatmaps: accuracy broken down by class
# =============================================================================
print("\nGenerating Chart 2: Per-model class accuracy heatmaps...")

class_cols = [c for c in df.columns if c not in ("Model", "Test_Dataset", "Overall_Accuracy")]

for model_label in model_order:

    df_model = df[df["Model"] == model_label].copy()

    if df_model.empty:
        print(f"  [SKIP] No data for model '{model_label}'.")
        continue

    df_model     = df_model.set_index("Test_Dataset")
    df_model     = df_model.reindex(dataset_order)
    heatmap_data = df_model[class_cols].T

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Accuracy (%)"},
        vmin=0,
        vmax=100,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(
        f"Accuracy Heatmap by Class — Model Trained on: {model_label}",
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Test Dataset (Error Rate)", fontweight="bold")
    ax.set_ylabel("Classes", fontweight="bold")

    plt.tight_layout()
    safe_label = model_label.replace(".", "_")
    save(fig, f"Heatmap_Model_{safe_label}")
    plt.close()


print("\nGenerating combined PDF with all charts...")

combined_pdf_path = os.path.join(CHARTS_OUTPUT_DIR, "ALL_CHARTS.pdf")

with PdfPages(combined_pdf_path) as pdf:

    # Page 1 — bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=df,
        x="Model",
        y="Overall_Accuracy",
        hue="Test_Dataset",
        hue_order=dataset_order,
        palette=paleta_semantica,
        edgecolor="black",
        linewidth=0.7,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", label_type="edge", fontsize=7, padding=2, rotation=90)
    if not baseline_rows.empty:
        ax.axhline(baseline, color="black", linewidth=1.2, linestyle="--", label=f"Baseline (non-approx / non-approx): {baseline:.1f}%")
    ax.set_title("Cross-Validation Performance: Training Context vs. Test Scenarios", fontweight="bold", pad=15)
    ax.set_xlabel("Training Context (Model)", fontweight="bold")
    ax.set_ylabel("Validation Accuracy (%)", fontweight="bold")
    ax.set_ylim(max(0, min_acc - 5), 112)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Validated On\n(Test Scenario):", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # Pages 2..N — one heatmap per model
    for model_label in model_order:
        df_model = df[df["Model"] == model_label].copy()
        if df_model.empty:
            continue
        df_model     = df_model.set_index("Test_Dataset").reindex(dataset_order)
        heatmap_data = df_model[class_cols].T

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu",
                    cbar_kws={"label": "Accuracy (%)"}, vmin=0, vmax=100, linewidths=0.5, ax=ax)
        ax.set_title(f"Accuracy Heatmap by Class — Model Trained on: {model_label}", fontweight="bold", pad=15)
        ax.set_xlabel("Test Dataset (Error Rate)", fontweight="bold")
        ax.set_ylabel("Classes", fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # PDF metadata
    meta = pdf.infodict()
    meta["Title"]   = "Cross-Validation Results — ResNet-50 Error Rate Study"
    meta["Subject"] = "Accuracy heatmaps and grouped bar charts"

print(f"  Saved combined PDF: {combined_pdf_path}")
print(f"\n✓ All charts generated and saved to: {CHARTS_OUTPUT_DIR}")
