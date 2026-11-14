import os
import re
import io
import csv
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm

from config import MODELS, DATASETS, CSV_OUTPUT

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_rgb(x):
    return x.convert("RGB")


# ══════════════════════════════════════════════
# BLOCK 1 — BINARY EXTRACTION (JPEG)
# ══════════════════════════════════════════════
PAYLOAD_MINIMO     = 1024
MARCADORES_VALIDOS = [b'\xff\xe0', b'\xff\xe1', b'\xff\xdb']
PADRAO_CRASH       = re.compile(br'z\s+[0-9a-fA-F]{16}\s+ra\s+[0-9a-fA-F]{16}')


def resgatar_jpeg_antes_do_crash(caminho_arquivo):
    try:
        if os.path.getsize(caminho_arquivo) == 0:
            return None, 'empty'

        with open(caminho_arquivo, 'rb') as f:
            conteudo = f.read()

        match_crash = PADRAO_CRASH.search(conteudo)

        if match_crash:
            dados_validos = conteudo[:match_crash.start()]
            teve_crash = True
        else:
            dados_validos = conteudo
            teve_crash = False

        inicio_jpeg = dados_validos.find(b'\xff\xd8')

        if inicio_jpeg == -1:
            return None, 'no_ffd8'

        payload = len(dados_validos) - inicio_jpeg - 2

        if payload < PAYLOAD_MINIMO:
            return None, 'corrompida_payload_pequeno'

        janela = dados_validos[inicio_jpeg+2:min(inicio_jpeg+22, len(dados_validos))]

        if not any(m in janela for m in MARCADORES_VALIDOS):
            return None, 'corrompida_sem_marcador'

        status = 'rescued_crash' if teve_crash else 'clean'

        return dados_validos[inicio_jpeg:], status

    except OSError:
        return None, 'os_error'


# ══════════════════════════════════════════════
# BLOCK 2 — RESILIENT DECODING
# ══════════════════════════════════════════════
def carregar_imagem_robusta(caminho_arquivo):

    if caminho_arquivo.lower().endswith((".tif", ".tiff")):
        try:
            img = Image.open(caminho_arquivo).convert("RGB")
            return img, "clean"
        except (OSError, SyntaxError, Image.UnidentifiedImageError):
            return None, "pil_rejected"

    bytes_imagem, status = resgatar_jpeg_antes_do_crash(caminho_arquivo)

    if bytes_imagem is None:
        return None, status

    try:
        img = Image.open(io.BytesIO(bytes_imagem)).convert("RGB")
        return img, status
    except (OSError, SyntaxError, Image.UnidentifiedImageError):
        return None, "pil_rejected"


# ══════════════════════════════════════════════
# BLOCK 3 — INVALID IMAGE COUNTER
# ══════════════════════════════════════════════
def count_invalid(dataset):

    stats = {
        'clean': 0,
        'rescued_crash': 0,
        'corrompida_payload_pequeno': 0,
        'corrompida_sem_marcador': 0,
        'no_ffd8': 0,
        'empty': 0,
        'pil_rejected': 0,
        'os_error': 0
    }

    for path, _ in dataset.samples:
        _, status = carregar_imagem_robusta(path)

        if status in stats:
            stats[status] += 1
        else:
            stats['pil_rejected'] += 1

    n_valid = stats['clean'] + stats['rescued_crash']
    n_invalid = sum(v for k, v in stats.items()
                    if k not in ('clean', 'rescued_crash'))

    return stats, n_valid, n_invalid


# ══════════════════════════════════════════════
# BLOCK 4 — ROBUST DATASET
# ══════════════════════════════════════════════
class RobustImageFolder(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):

        path, label = self.samples[index]

        img, _ = carregar_imagem_robusta(path)

        if img is None:
            return None

        if self.transform:
            img = self.transform(img)

        return img, label


def safe_collate(batch):
    batch = [b for b in batch if b is not None]

    if not batch:
        return torch.Tensor(), torch.Tensor()

    return torch.utils.data.dataloader.default_collate(batch)


# ══════════════════════════════════════════════
# BLOCK 5 — TRANSFORMS
# ══════════════════════════════════════════════
transforms_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(to_rgb),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":

    print(f"\nUsing device: {device}")
    print(f"Models   : {list(MODELS.keys())}")
    print(f"Datasets : {list(DATASETS.keys())}")
    print(f"Output   : {CSV_OUTPUT}\n")

    header_written = False

    for model_label, model_path in MODELS.items():

        if not os.path.exists(model_path):
            print(f"[SKIP] Model not found: {model_path}")
            continue

        print(f"\n[{'='*50}]")
        print(f"  Loading model: [{model_label}]")
        print(f"[{'='*50}]")

        model = torch.load(
            model_path,
            map_location=device,
            weights_only=False
        )

        model.eval()

        for data_label, dataset_path in DATASETS.items():

            if not os.path.exists(dataset_path):
                print(f"[SKIP] Dataset not found: {dataset_path}")
                continue

            print(f"\nEvaluating on dataset: [{data_label}]")

            dataset = RobustImageFolder(
                root=dataset_path,
                transform=transforms_eval
            )

            classes_dataset = dataset.classes

            print("Checking dataset integrity...")

            stats, n_valid, n_invalid = count_invalid(dataset)

            print(f"clean         : {stats['clean']}")
            print(f"rescued_crash : {stats['rescued_crash']}")
            print(f"discarded     : {n_invalid}")
            print(f"total valid   : {n_valid}")

            if not header_written:
                with open(CSV_OUTPUT, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["Model", "Test_Dataset", "Overall_Accuracy"]
                        + classes_dataset
                    )
                header_written = True

            loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=safe_collate
            )

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in tqdm(loader,
                                           desc="Inference",
                                           leave=False):

                    if inputs.numel() == 0:
                        continue

                    inputs = inputs.to(device)

                    outputs = model(inputs)

                    _, predicted = outputs.max(1)

                    all_preds.extend(predicted.cpu().tolist())
                    all_labels.extend(labels.tolist())

            if len(all_labels) == 0:
                print(f"[WARNING] No valid images in {data_label}")
                continue

            correct = sum(
                p == l for p, l in zip(all_preds, all_labels)
            )

            acc_total = 100.0 * correct / len(all_labels)

            class_correct = [0] * len(classes_dataset)
            class_total = [0] * len(classes_dataset)

            for p, l in zip(all_preds, all_labels):
                class_total[l] += 1
                if p == l:
                    class_correct[l] += 1

            acc_per_class = []

            for i in range(len(classes_dataset)):
                if class_total[i] > 0:
                    acc_per_class.append(
                        100.0 * class_correct[i] / class_total[i]
                    )
                else:
                    acc_per_class.append(0.0)

            print(f"Overall accuracy: {acc_total:.2f}%")

            with open(CSV_OUTPUT, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [model_label, data_label, round(acc_total, 2)]
                    + [round(x, 2) for x in acc_per_class]
                )

    print(f"\n✓ Batch validation complete! Results saved to: {CSV_OUTPUT}")
