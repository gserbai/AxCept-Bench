import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv


def to_rgb(x):
    return x.convert("RGB")


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


# ──────────────────────────────────────────────
# Configuração do experimento
# ──────────────────────────────────────────────
EXPERIMENT_NAME = "non-approx"

LOG_FILE   = f"log_{EXPERIMENT_NAME}.csv"
MODEL_FILE = f"R50_model_{EXPERIMENT_NAME}.pth"

TRAIN_DIR  = "/home/guilherme/Pictures/Datas/non-approx/train"
TEST_DIR   = "/home/guilherme/Pictures/Datas/non-approx/test"

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs     = 50

print(f"Dispositivo: {device}")


# ──────────────────────────────────────────────
# Transformações
# ──────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.Lambda(to_rgb),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(to_rgb),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
print("Carregando dataset TIFF...")

train_dataset = ImageFolder(
    root=TRAIN_DIR,
    transform=train_transforms
)

test_dataset = ImageFolder(
    root=TEST_DIR,
    transform=test_transforms
)

print(f"Treino: {len(train_dataset)} imagens")
print(f"Teste : {len(test_dataset)} imagens")
print(f"Classes: {len(train_dataset.classes)}")


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
    worker_init_fn=seed_worker,
    generator=g
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ──────────────────────────────────────────────
# Modelo
# ──────────────────────────────────────────────
model = resnet50(
    weights=ResNet50_Weights.IMAGENET1K_V1
)

model.fc = nn.Linear(
    model.fc.in_features,
    21
)

model = model.to(device)


# ──────────────────────────────────────────────
# Loss / Optimizer / Scheduler
# ──────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=15,
    gamma=0.1
)


# ──────────────────────────────────────────────
# Treino
# ──────────────────────────────────────────────
def train(model, dataloader, optimizer, criterion):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Treinando"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        del outputs, loss
        torch.cuda.empty_cache()

    return running_loss / len(dataloader), 100.0 * correct / total


# ──────────────────────────────────────────────
# Validação
# ──────────────────────────────────────────────
def validate(model, dataloader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validando"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


# ──────────────────────────────────────────────
# CSV header
# ──────────────────────────────────────────────
write_header = not os.path.exists(LOG_FILE)

with open(LOG_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    if write_header:
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "is_best"
        ])


# ──────────────────────────────────────────────
# Loop principal
# ──────────────────────────────────────────────
best_val_acc = 0.0

print("\nIniciando treinamento...\n")

torch.cuda.empty_cache()
for epoch in range(epochs):

    train_loss, train_acc = train(
        model,
        train_loader,
        optimizer,
        criterion
    )

    val_loss, val_acc = validate(
        model,
        test_loader,
        criterion
    )

    scheduler.step()

    is_best = val_acc > best_val_acc

    if is_best:
        best_val_acc = val_acc
        torch.save(model, MODEL_FILE)
        print(f"  ✓ Melhor modelo salvo: {val_acc:.2f}%")

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            int(is_best)
        ])

print("\nTreinamento finalizado.")
print(f"Melhor acurácia: {best_val_acc:.2f}%")
print(f"Modelo salvo em: {MODEL_FILE}")
