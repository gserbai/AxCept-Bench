import os
import re
import io
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image, ImageFile
import csv

# ──────────────────────────────────────────────
# Permite que o PIL carregue JPEGs truncados
# (MCUs incompletos recebem padding estático)
# ──────────────────────────────────────────────
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ──────────────────────────────────────────────
# Reprodutibilidade
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ══════════════════════════════════════════════
# BLOCO 1 — BINARY EXTRACTION & VALIDATION
# ══════════════════════════════════════════════
PAYLOAD_MINIMO   = 1024
MARCADORES_VALIDOS = [b'\xff\xe0', b'\xff\xe1', b'\xff\xdb']

# Padrão exato do log de crash do AxPike RISC-V
PADRAO_CRASH = re.compile(br'z\s+[0-9a-fA-F]{16}\s+ra\s+[0-9a-fA-F]{16}')

def resgatar_jpeg_antes_do_crash(caminho_arquivo):
    """
    Lê o arquivo em modo binário. Se detectar o log de crash do simulador
    RISC-V, trunca o stream naquele ponto. Valida se o payload restante é
    um JPEG estruturalmente mínimo (FFD8 + marcador + ≥ 1 KB de payload).

    Retorna:
        (bytes_imagem, status)  — status: 'clean' | 'rescued_crash' | motivo_descarte
        (None,         status)  — quando o arquivo deve ser descartado
    """
    try:
        if os.path.getsize(caminho_arquivo) == 0:
            return None, 'empty'

        with open(caminho_arquivo, 'rb') as f:
            conteudo = f.read()

        # ── Interrompe o stream no crash signature ──
        match_crash = PADRAO_CRASH.search(conteudo)
        if match_crash:
            dados_validos = conteudo[:match_crash.start()]
            teve_crash    = True
        else:
            dados_validos = conteudo
            teve_crash    = False

        # ── Localiza o SOI marker (FFD8) ──
        inicio_jpeg = dados_validos.find(b'\xff\xd8')
        if inicio_jpeg == -1:
            return None, 'no_ffd8'

        # ── Valida tamanho mínimo do payload ──
        payload = len(dados_validos) - inicio_jpeg - 2
        if payload < PAYLOAD_MINIMO:
            return None, 'corrompida_payload_pequeno'

        # ── Valida presença de marcadores (JFIF / EXIF / DQT) ──
        janela = dados_validos[inicio_jpeg + 2 : min(inicio_jpeg + 22, len(dados_validos))]
        if not any(m in janela for m in MARCADORES_VALIDOS):
            return None, 'corrompida_sem_marcador'

        status = 'rescued_crash' if teve_crash else 'clean'
        return dados_validos[inicio_jpeg:], status

    except OSError:
        return None, 'os_error'


# ══════════════════════════════════════════════
# BLOCO 2 — RESILIENT DECODING
# Carrega o stream resgatado direto da RAM
# ══════════════════════════════════════════════
def carregar_imagem_robusta(caminho_arquivo):
    """
    Tenta carregar a imagem aplicando a pipeline completa:
      1. Detecção + truncamento no crash signature (Binary Extraction)
      2. Decodificação resiliente via BytesIO (Resilient Decoding)
         — regiões ausentes (MCUs incompletos) recebem padding estático
           gerado pelo Pillow com LOAD_TRUNCATED_IMAGES = True

    Retorna:
        (PIL.Image, status)  — imagem válida + status binário
        (None,      status)  — arquivo descartado + motivo
    """
    bytes_imagem, status = resgatar_jpeg_antes_do_crash(caminho_arquivo)

    if bytes_imagem is None:
        return None, status

    try:
        img = Image.open(io.BytesIO(bytes_imagem)).convert('RGB')
        return img, status
    except (OSError, SyntaxError, Image.UnidentifiedImageError):
        return None, 'pil_rejected'


# ══════════════════════════════════════════════
# BLOCO 3 — DATASET ROBUSTO
# resgate binário + decodificação resiliente
# ══════════════════════════════════════════════
class RobustImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]

        img, status = carregar_imagem_robusta(path)

        if img is None:
            return None   # descartado pelo safe_collate

        if self.transform:
            img = self.transform(img)

        return img, label


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)


# ══════════════════════════════════════════════
# BLOCO 4 — CONTAGEM ANTECIPADA DE INVÁLIDOS
# ══════════════════════════════════════════════
def count_invalid(dataset):
    """
    Percorre o dataset uma vez antes do treino e classifica cada arquivo
    usando a mesma pipeline de resgate do ssmiucmerced.py.
    """
    stats = {
        'clean':                      0,
        'rescued_crash':              0,
        'corrompida_payload_pequeno': 0,
        'corrompida_sem_marcador':    0,
        'no_ffd8':                    0,
        'empty':                      0,
        'pil_rejected':               0,
        'os_error':                   0,
    }

    for path, _ in dataset.samples:
        _, status = carregar_imagem_robusta(path)
        if status in stats:
            stats[status] += 1
        else:
            stats['pil_rejected'] += 1   # status inesperado → descarta

    validas   = stats['clean'] + stats['rescued_crash']
    invalidas = sum(v for k, v in stats.items() if k not in ('clean', 'rescued_crash'))

    return stats, validas, invalidas


# ──────────────────────────────────────────────
# Configurações do experimento
# ↓ muda só aqui para trocar de cenário ↓
# ──────────────────────────────────────────────
ERROR_RATE = "1e-5"

LOG_FILE   = f"log_{ERROR_RATE}.csv"
MODEL_FILE = f"R50_model_error_rate_{ERROR_RATE}.pth"
BASE_DIR   = f"/home/guilherme/Pictures/UC_MERCED_LAND_USE/DataSets/dataset_error_rate_{ERROR_RATE}/src/dataset_error_rate_{ERROR_RATE}"
TRAIN_DIR  = f"{BASE_DIR}/train"
TEST_DIR   = f"{BASE_DIR}/test"

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs     = 50

# ──────────────────────────────────────────────
# Transformações
# ──────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ──────────────────────────────────────────────
# Dataset e DataLoader
# ──────────────────────────────────────────────
train_dataset = RobustImageFolder(root=TRAIN_DIR, transform=train_transforms)
test_dataset  = RobustImageFolder(root=TEST_DIR,  transform=test_transforms)

# ──────────────────────────────────────────────
# Contagem antecipada — roda uma vez antes do treino
# ──────────────────────────────────────────────
print("Verificando integridade do dataset (Binary Extraction + Resilient Decoding)...")

train_stats, train_validas, train_invalidas = count_invalid(train_dataset)
test_stats,  test_validas,  test_invalidas  = count_invalid(test_dataset)

total_validas        = train_validas   + test_validas
total_invalidas      = train_invalidas + test_invalidas
total_rescued_crash  = train_stats['rescued_crash'] + test_stats['rescued_crash']
total_clean          = train_stats['clean']          + test_stats['clean']

print(f"\n{'='*60}")
print("  RELATÓRIO DE INTEGRIDADE DO DATASET")
print(f"{'='*60}")
print(f"  [TREINO]")
print(f"    Clean                 : {train_stats['clean']}")
print(f"    Resgatadas (Crash)    : \033[96m{train_stats['rescued_crash']}\033[0m")
print(f"    Payload < 1KB         : {train_stats['corrompida_payload_pequeno']}")
print(f"    Sem Marcador JFIF/DQT : {train_stats['corrompida_sem_marcador']}")
print(f"    Sem FFD8              : {train_stats['no_ffd8']}")
print(f"    PIL Rejeitou          : {train_stats['pil_rejected']}")
print(f"    Vazias (0B)           : {train_stats['empty']}")
print(f"  [TESTE]")
print(f"    Clean                 : {test_stats['clean']}")
print(f"    Resgatadas (Crash)    : \033[96m{test_stats['rescued_crash']}\033[0m")
print(f"    Payload < 1KB         : {test_stats['corrompida_payload_pequeno']}")
print(f"    Sem Marcador JFIF/DQT : {test_stats['corrompida_sem_marcador']}")
print(f"    Sem FFD8              : {test_stats['no_ffd8']}")
print(f"    PIL Rejeitou          : {test_stats['pil_rejected']}")
print(f"    Vazias (0B)           : {test_stats['empty']}")
print(f"{'='*60}")
print(f"  Total válidas (SSIM-equiv): \033[92m{total_validas}\033[0m  (clean: {total_clean} | rescued: {total_rescued_crash})")
print(f"  Total descartadas         : \033[93m{total_invalidas}\033[0m")
print(f"{'='*60}\n")

# ──────────────────────────────────────────────
# DataLoader com seed nos workers
# ──────────────────────────────────────────────
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=safe_collate,
    worker_init_fn=seed_worker,
    generator=g
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=safe_collate
)

# ──────────────────────────────────────────────
# Modelo
# ──────────────────────────────────────────────
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 21)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# ──────────────────────────────────────────────
# Otimizador e Scheduler
# ──────────────────────────────────────────────
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# ──────────────────────────────────────────────
# Treinamento
# ──────────────────────────────────────────────
def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0
    for inputs, labels in tqdm(dataloader):
        if inputs.numel() == 0:
            continue
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        del outputs, loss
        torch.cuda.empty_cache()
    return running_loss / len(dataloader), 100. * correct / total

# ──────────────────────────────────────────────
# Validação
# ──────────────────────────────────────────────
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            if inputs.numel() == 0:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted  = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), 100. * correct / total

# ──────────────────────────────────────────────
# Loop principal
# ──────────────────────────────────────────────
best_val_acc   = 0.0
best_model_wts = model.state_dict()

write_header = not os.path.exists(LOG_FILE)
with open(LOG_FILE, 'a', newline='') as f:
    if write_header:
        csv.writer(f).writerow([
            'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'total_validas', 'total_clean', 'total_rescued_crash',
            'total_descartadas', 'is_best'
        ])

torch.cuda.empty_cache()
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss,   val_acc   = validate(model, test_loader, criterion)
    scheduler.step()

    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc   = val_acc
        best_model_wts = model.state_dict()
        torch.save(model, MODEL_FILE)
        print(f"  ✓ Melhor modelo salvo — Epoch {epoch+1} | Val Acc: {val_acc:.2f}%")

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoch + 1, train_loss, train_acc, val_loss, val_acc,
            total_validas, total_clean, total_rescued_crash,
            total_invalidas, int(is_best)
        ])
