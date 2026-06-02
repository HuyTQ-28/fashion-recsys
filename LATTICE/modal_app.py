from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).parent
FREEDOM_HM_DIR = PROJECT_ROOT.parent / "FREEDOM" / "hm"
REMOTE_ROOT = "/app"
REMOTE_CODES = f"{REMOTE_ROOT}/codes"
REMOTE_DATA_ROOT = f"{REMOTE_ROOT}/FREEDOM"
OUTPUT_DIR = "/outputs"

A100_TRAIN_BATCH_SIZE = 32768
A100_EVAL_BATCH_SIZE = 8192
A100_EVAL_CORES = 8

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirement.txt"))
    .workdir(REMOTE_CODES)
    .env(
        {
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "NUMEXPR_MAX_THREADS": "8",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(PROJECT_ROOT / "codes", REMOTE_CODES)
    .add_local_dir(FREEDOM_HM_DIR, f"{REMOTE_DATA_ROOT}/hm")
)

output_volume = modal.Volume.from_name("lattice-runs", create_if_missing=True)
app = modal.App("lattice-train", image=image)


@app.function(
    gpu="A100-40GB",
    cpu=8,
    memory=98304,
    timeout=12 * 60 * 60,
    volumes={OUTPUT_DIR: output_volume},
)
def train_remote(
    dataset: str = "hm",
    epochs: int = 200,
    batch_size: int = A100_TRAIN_BATCH_SIZE,
    eval_batch_size: int = A100_EVAL_BATCH_SIZE,
    eval_cores: int = A100_EVAL_CORES,
    lr: float = 0.0005,
    gpu_id: int = 0,
    topk: int = 10,
    verbose: int = 5,
    early_stopping_patience: int = 10,
    cf_model: str = "lightgcn",
):
    import os
    import runpy
    import shutil
    import sys
    from pathlib import Path

    import torch

    os.chdir(REMOTE_CODES)
    sys.path.insert(0, REMOTE_CODES)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    sys.argv = [
        "main.py",
        "--data_path",
        REMOTE_DATA_ROOT,
        "--dataset",
        dataset,
        "--epoch",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--eval_batch_size",
        str(eval_batch_size),
        "--eval_cores",
        str(eval_cores),
        "--lr",
        str(lr),
        "--gpu_id",
        str(gpu_id),
        "--topk",
        str(topk),
        "--verbose",
        str(verbose),
        "--early_stopping_patience",
        str(early_stopping_patience),
        "--cf_model",
        cf_model,
    ]

    runpy.run_path("main.py", run_name="__main__")

    output_root = Path(OUTPUT_DIR)
    cache_root = output_root / "hm_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(REMOTE_DATA_ROOT) / dataset
    for pattern in ("*.npz", "*_adj_*.pt"):
        for src in data_dir.glob(pattern):
            shutil.copy2(src, cache_root / src.name)

    output_volume.commit()
    return "Finished LATTICE training on Modal. Cached graph files are in Modal volume 'lattice-runs'."


@app.local_entrypoint()
def main(
    dataset: str = "hm",
    epochs: int = 200,
    batch_size: int = A100_TRAIN_BATCH_SIZE,
    eval_batch_size: int = A100_EVAL_BATCH_SIZE,
    eval_cores: int = A100_EVAL_CORES,
    lr: float = 0.0005,
    gpu_id: int = 0,
    topk: int = 10,
    verbose: int = 5,
    early_stopping_patience: int = 10,
    cf_model: str = "lightgcn",
):
    call = train_remote.spawn(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        eval_cores=eval_cores,
        lr=lr,
        gpu_id=gpu_id,
        topk=topk,
        verbose=verbose,
        early_stopping_patience=early_stopping_patience,
        cf_model=cf_model,
    )
    print(f"Started Modal training call: {call.object_id}")
    print(call.get())
