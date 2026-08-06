# ============================================================
#  config.py — Edit this file before running val.py or geradorgraficonn.py
#  Just fill in the paths below. Nothing else needs to change.
# ============================================================

# --- Models ---
# Add one entry per model: "label": "full/path/to/model.pth"
MODELS = {
    "non-approx": "/home/guilherme/Pictures/R50_model_non-approx.pth",
    "1e-5":       "/home/guilherme/Pictures/R50_model_error_rate_1e-5.pth",
    "1e-4":       "/home/guilherme/Pictures/R50_model_error_rate_1e-4.pth",
    "1e-3":       "/home/guilherme/Pictures/R50_model_error_rate_1e-3.pth",
    "1e-2":       "/home/guilherme/Pictures/R50_model_error_rate_1e-2.pth",
    "1e-1":       "/home/guilherme/Pictures/R50_model_error_rate_1e-1.pth",
}

# --- Datasets ---
# Add one entry per dataset: "label": "full/path/to/test/folder"
# The folder must contain one sub-folder per class.
DATASETS = {
    "non-approx": "/home/guilherme/Pictures/Datas/non-approx/test",
    "1e-5":       "/home/guilherme/Pictures/Datas/dataset_error_rate_1e-5/test",
    "1e-4":       "/home/guilherme/Pictures/Datas/dataset_error_rate_1e-4/test",
    "1e-3":       "/home/guilherme/Pictures/Datas/dataset_error_rate_1e-3/test",
    "1e-2":       "/home/guilherme/Pictures/Datas/dataset_error_rate_1e-2/test",
    "1e-1":       "/home/guilherme/Pictures/Datas/dataset_error_rate_1e-1/test",
}

# --- Output CSV ---
# Full path where the validation results file will be saved.
CSV_OUTPUT = "/home/guilherme/Pictures/resultados_consolidados.csv"

# --- Output charts folder ---
# Folder where all generated charts will be saved.
# It will be created automatically if it does not exist.
CHARTS_OUTPUT_DIR = "/home/guilherme/Pictures/graficos_paper"
