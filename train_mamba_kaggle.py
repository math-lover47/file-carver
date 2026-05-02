# %% [markdown]
# # Mamba File Carver - Training Pipeline
# This notebook trains a Mamba-based file carver using NapierOne-Lean dataset.
# It supports Multi-GPU (DDP), Optuna Hyperparameter Search, and Weights & Biases (W&B) logging.

# %%

# !python --version

# %%
# !pip uninstall -y mamba-ssm causal-conv1d torch torchvision torchaudio

# %%
# !pip install uv wandb
# %%
# !uv pip install --system torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128

# %%
# !uv pip install --system https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# %%
# !uv pip install --system https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# %%
# 1. Imports and W&B Login
import mamba_ssm
import os
import gc
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import MambaModel, MambaConfig
import optuna
from tqdm import tqdm
import wandb

# --- W&B Authentication ---
# Вход в Weights & Biases

# %% [markdown]
# # 2. Configuration & Hyperparameters
# Global settings for training and the dataset paths.

# %%
DATASET_ROOT = "/kaggle/working/napier_micro"
BLOCK_SIZE = 512

# Training settings
BATCH_SIZE_TRAIN = 32
EPOCHS = 10
D_MODEL = 768
NUM_CLASSES = 5
N_LAYERS = 12
REID_DIM = 256

# Optuna settings
OPTUNA_SEARCH = False
BATCH_SIZE_OPTUNA = 8
N_TRIALS = 10

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# PATCH: Compatibility between mamba-ssm 2.x and transformers

# 1. Импортируем функции напрямую из их реальных мест в версии 2.x
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    # В версии 2.x selective_state_update может называться иначе или лежать в другом месте
    try:
        from mamba_ssm.ops.selective_state_update import selective_state_update
    except ImportError:
        # Если в ops его нет, создаем "заглушку" или пытаемся найти альтернативу
        # Но обычно он там есть, просто не экспортирован в корень
        selective_state_update = None

    # 2. Жёстко прописываем их в модуль mamba_ssm
    # Именно эти атрибуты ищет transformers/models/mamba/modeling_mamba.py на строке 210
    mamba_ssm.selective_scan_fn = selective_scan_fn
    mamba_ssm.mamba_inner_fn = mamba_inner_fn

    if selective_state_update is not None:
        mamba_ssm.selective_state_update = selective_state_update
    else:
        # Если его совсем нет, создаем пустую функцию, чтобы не падал AttributeError
        # Transformers использует её только для генерации (inference), для обучения хватит mamba_inner_fn
        def dummy_update(*args, **kwargs): return None
        mamba_ssm.selective_state_update = dummy_update
        if __name__ == "__main__":
            print("⚠️ selective_state_update not found, using dummy (fine for training)")

    if __name__ == "__main__":
        print("🚀 Mamba-ssm successfully patched for Transformers!")

except ImportError as e:
    print(
        f"❌ Ошибка импорта: {e}. Проверь, что установка через uv прошла успешно.")

# Dynamic Num Classes Loading
try:
    with open(os.path.join(DATASET_ROOT, "classes_micro.json"), 'r') as f:
        NUM_CLASSES = len(json.load(f))
except Exception:
    NUM_CLASSES = 41

# %% [markdown]
# # 3. Dataset Definition
# Loading the `.npy` files created during the Lean dataset preparation.

# %%


class NapierMambaDataset(Dataset):
    def __init__(self, x_path, y_path, l_path):
        self.X = np.load(x_path, mmap_mode='r')
        self.Y = np.load(y_path)
        self.L = np.load(l_path)
        self.num_classes = len(np.unique(self.Y))

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].astype(np.int64))
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        # Boundary logic: 1.0 if it's the first block, 0.0 otherwise
        bound = torch.tensor(
            [1.0 if self.L[idx, 1] == 0 else 0.0, 0.0], dtype=torch.float32)
        file_id = torch.tensor(self.L[idx, 0], dtype=torch.long)
        return x, y, bound, file_id

# %% [markdown]
# # 4. Model Architecture (MambaTriHead)
# The Mamba backbone combined with three classification heads: File Type, Boundary, and Re-ID.

# %%


class MambaTriHead(nn.Module):
    def __init__(self, d_model, n_layers, num_classes, reid_dim):
        super().__init__()
        config = MambaConfig(
            d_model=d_model,
            n_layer=n_layers,
            vocab_size=257,
            intermediate_size=2 * d_model,
            use_cache=False
        )
        self.mamba = MambaModel(config)
        self.mamba.embeddings = nn.Embedding(257, d_model)
        # Отключаем gradient checkpointing: для 12 слоев и батча 16 он не нужен,
        # но при этом вызывает баг несовпадения тензоров при recomputation.
        # self.mamba.gradient_checkpointing_enable()

        self.head_class = nn.Linear(d_model, num_classes)
        self.head_bound = nn.Linear(d_model, 2)
        self.head_reid = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, reid_dim)
        )

    def forward(self, input_ids):
        outputs = self.mamba(input_ids)
        # Use the last token for causal sequence classification
        pooled = outputs.last_hidden_state[:, -1, :]

        p = pooled.float()
        logits_class = self.head_class(p)
        logits_bound = self.head_bound(p)
        reid_embeds = F.normalize(self.head_reid(p), p=2, dim=1)

        return logits_class, logits_bound, reid_embeds

# %% [markdown]
# # 5. Training Utilities & Triplet Loss
# Helper functions for calculating losses and running a single training epoch.

# %%


def batch_hard_triplet_loss(embeddings, labels, margin=1.0):
    """Calculates the triplet loss in FP32 with NaN protections."""
    embeddings = embeddings.float()
    dist_mat = torch.cdist(embeddings, embeddings, p=2) + 1e-8

    mask_pos = (labels.view(-1, 1) == labels.view(1, -1)).float() - \
        torch.eye(labels.size(0), device=labels.device)
    mask_neg = (labels.view(-1, 1) != labels.view(1, -1)).float()

    loss, valid = 0.0, 0
    for i in range(embeddings.size(0)):
        pos = dist_mat[i][mask_pos[i] == 1]
        neg = dist_mat[i][mask_neg[i] == 1]
        if len(pos) > 0 and len(neg) > 0:
            loss += F.relu(pos.max() - neg.min() + margin)
            valid += 1

    return loss / valid if valid > 0 else embeddings.sum() * 0.0


def train_epoch(model, dataloader, optimizer, scaler, weights, rank, epoch=0, scheduler=None, use_wandb=False):
    model.train()
    pbar = tqdm(
        dataloader, desc=f"GPU {rank} | Epoch {epoch}", disable=(rank != 0))
    total_loss, total_acc = 0, 0
    valid_batches = 0

    for step, (x, y, b, fid) in enumerate(pbar):
        x, y, b, fid = x.cuda(rank), y.cuda(rank), b.cuda(rank), fid.cuda(rank)
        optimizer.zero_grad()

        lc, lb, em = model(x)
        loss_c = F.cross_entropy(lc, y)
        loss_b = F.binary_cross_entropy_with_logits(lb, b)
        loss_r = batch_hard_triplet_loss(em, fid)
        loss = weights['type']*loss_c + weights['bound']*loss_b + weights['reid']*loss_r

        loss_is_nan = torch.tensor([1 if torch.isnan(loss) else 0], dtype=torch.long, device=x.device)
        dist.all_reduce(loss_is_nan, op=dist.ReduceOp.SUM)
        if loss_is_nan.item() > 0:
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        if scheduler:
            scheduler.step()

        # Metrics
        loss_val = loss.item()
        acc_val = (lc.argmax(1) == y).float().mean().item()
        total_loss += loss_val
        total_acc += acc_val
        valid_batches += 1

        if rank == 0:
            pbar.set_postfix(
                {"Loss": f"{loss_val:.3f}", "Acc": f"{acc_val:.2f}"})
            if use_wandb and step % 10 == 0:
                wandb.log({"step_loss": loss_val, "step_acc": acc_val,
                          "learning_rate": optimizer.param_groups[0]['lr']})

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_acc = total_acc / valid_batches if valid_batches > 0 else 0.0
    return avg_loss, avg_acc


@torch.no_grad()
def validate_epoch(model, dataloader, weights, rank):
    model.eval()
    pbar = tqdm(
        dataloader, desc=f"GPU {rank} | Validation", disable=(rank != 0))
    total_loss, total_acc = 0, 0
    valid_batches = 0

    for x, y, b, fid in pbar:
        x, y, b, fid = x.cuda(rank), y.cuda(rank), b.cuda(rank), fid.cuda(rank)

        lc, lb, em = model(x)
        loss_c = F.cross_entropy(lc, y)
        loss_b = F.binary_cross_entropy_with_logits(lb, b)
        loss_r = batch_hard_triplet_loss(em, fid)
        loss = weights['type']*loss_c + weights['bound']*loss_b + weights['reid']*loss_r

        if torch.isnan(loss):
            continue

        loss_val = loss.item()
        acc_val = (lc.argmax(1) == y).float().mean().item()
        total_loss += loss_val
        total_acc += acc_val
        valid_batches += 1

        if rank == 0:
            pbar.set_postfix(
                {"Loss": f"{loss_val:.3f}", "Acc": f"{acc_val:.2f}"})

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_acc = total_acc / valid_batches if valid_batches > 0 else 0.0
    return avg_loss, avg_acc

# %% [markdown]
# # 6. Distributed Training Setup (DDP)
# Process group initialization for Multi-GPU training.

# %%


def setup_ddp(rank, world_size):
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# %% [markdown]
# # 7. Optuna Objective
# Hyperparameter search function.

# %%


def objective(trial):
    rank, world_size = 0, 1
    setup_ddp(rank, world_size)

    lr = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
    weights = {
        'type': trial.suggest_float("lt", 0.5, 1.0),
        'bound': trial.suggest_float("lb", 0.1, 0.5),
        'reid': trial.suggest_float("l_reid", 0.1, 0.5)
    }

    dataset = NapierMambaDataset(
        os.path.join(DATASET_ROOT, "napier_X_micro.npy"),
        os.path.join(DATASET_ROOT, "napier_Y_micro.npy"),
        os.path.join(DATASET_ROOT, "napier_L_micro.npy")
    )
    # 2% subset for fast Optuna trials
    subset = Subset(dataset, np.arange(0, int(len(dataset)*0.02)))
    loader = DataLoader(subset, batch_size=BATCH_SIZE_OPTUNA,
                        shuffle=True, num_workers=2)

    model = MambaTriHead(D_MODEL, N_LAYERS, NUM_CLASSES, REID_DIM).cuda(rank)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')

    try:
        avg_loss, _ = train_epoch(model, loader, opt, scaler, weights, rank)
    finally:
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()

    return avg_loss if not np.isnan(avg_loss) else 1.0

# %% [markdown]
# # 8. Main Training Worker
# The main distributed training loop with W&B logging.

# %%


def main_worker(rank, world_size, best_weights, best_lr):
    setup_ddp(rank, world_size)

    if rank == 0:
        wandb.init(
            entity="math-lover47-aitu",
            project="custom-models",
            name="mamba_carver_final",
            config={
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "batch_size": BATCH_SIZE_TRAIN,
                "epochs": EPOCHS,
                "lr": best_lr,
                "weights": best_weights
            }
        )

    dataset = NapierMambaDataset(
        os.path.join(DATASET_ROOT, "napier_X_micro.npy"),
        os.path.join(DATASET_ROOT, "napier_Y_micro.npy"),
        os.path.join(DATASET_ROOT, "napier_L_micro.npy")
    )

    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_sampler = DistributedSampler(train_ds, world_size, rank)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN // world_size,
                              sampler=train_sampler, num_workers=4)

    val_sampler = DistributedSampler(val_ds, world_size, rank)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_TRAIN // world_size,
                            sampler=val_sampler, num_workers=4)

    model = MambaTriHead(D_MODEL, N_LAYERS, NUM_CLASSES, REID_DIM).cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=best_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=best_lr*2, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_epoch(
            model, train_loader, opt, None, best_weights, rank,
            epoch=epoch, scheduler=sched, use_wandb=(rank == 0)
        )

        val_loss, val_acc = validate_epoch(
            model, val_loader, best_weights, rank
        )

        if rank == 0:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.module.state_dict(), "mamba_carver_best.pth")
                print(f"🔥 New best model saved! Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(
                    f"⚠️ Early stopping counter: {patience_counter}/{patience}")

            torch.save(model.module.state_dict(), "mamba_carver_last.pth")

        counter_tensor = torch.tensor(
            [patience_counter], dtype=torch.long).cuda(rank)
        dist.broadcast(counter_tensor, src=0)
        patience_counter = counter_tensor.item()

        if patience_counter >= patience:
            if rank == 0:
                print("🛑 Early stopping triggered!")
            break

    dist.barrier()
    model.module.load_state_dict(torch.load(
        "mamba_carver_best.pth", weights_only=True))

    test_sampler = DistributedSampler(test_ds, world_size, rank)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_TRAIN //
                             world_size, sampler=test_sampler, num_workers=4)
    test_loss, test_acc = validate_epoch(
        model, test_loader, best_weights, rank)

    if rank == 0:
        wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        print(f"📊 Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        wandb.finish()

    dist.destroy_process_group()

# %% [markdown]
# # 9. Execution Entry Point


# %%
if __name__ == "__main__":
    wandb.login(
        key="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")
    if OPTUNA_SEARCH:
        print("Starting Optuna Hyperparameter Search...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS)
        print("Best params:", study.best_params)

        best_lr = study.best_params['lr']
        best_weights = {
            'type': study.best_params['lt'], 'bound': study.best_params['lb'], 'reid': study.best_params['l_reid']}
    else:
        best_lr = 4.2295330706147926e-05
        best_weights = {
            'type': 0.52812425100632,
            'bound': 0.4945280652869767,
            'reid': 0.10346203551577036
        }

    print(f"Starting Final Training...")
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, best_weights,
             best_lr), nprocs=world_size, join=True)
