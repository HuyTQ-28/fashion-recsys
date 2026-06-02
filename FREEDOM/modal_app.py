from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).parent
REMOTE_ROOT = "/app"
OUTPUT_DIR = "/outputs"
A100_TRAIN_BATCH_SIZE = 32768
A100_EVAL_BATCH_SIZE = 16384

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirement.txt"))
    .workdir(f"{REMOTE_ROOT}/src")
    .env(
        {
            "NUMEXPR_MAX_THREADS": "8",
            "OMP_NUM_THREADS": "8",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(PROJECT_ROOT / "src", f"{REMOTE_ROOT}/src")
    .add_local_dir(PROJECT_ROOT / "hm", f"{REMOTE_ROOT}/hm")
)

output_volume = modal.Volume.from_name("freedom-runs", create_if_missing=True)
app = modal.App("freedom-train", image=image)


@app.function(
    gpu="A100-40GB",
    cpu=8,
    memory=65536,
    timeout=12 * 60 * 60,
    volumes={OUTPUT_DIR: output_volume},
)
def train_remote(
    model: str = "FREEDOM",
    dataset: str = "hm",
    epochs: int = 0,
    stopping_step: int = 20,
    early_stopping: bool = True,
    early_stopping_min_delta: float = 0.0,
    train_batch_size: int = A100_TRAIN_BATCH_SIZE,
    eval_batch_size: int = A100_EVAL_BATCH_SIZE,
    save_model: bool = True,
):
    import os
    import shutil
    import sys
    from pathlib import Path

    os.chdir(f"{REMOTE_ROOT}/src")
    sys.path.insert(0, REMOTE_ROOT)
    sys.path.insert(0, f"{REMOTE_ROOT}/src")

    from src.main import run_training
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    config_dict = {
        "gpu_id": 0,
        "use_gpu": True,
        "stopping_step": stopping_step,
        "early_stopping": early_stopping,
        "early_stopping_min_delta": early_stopping_min_delta,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
    }
    if epochs > 0:
        config_dict["epochs"] = epochs

    run_training(
        model=model,
        dataset=dataset,
        config_dict=config_dict,
        save_model=save_model,
    )

    output_root = Path(OUTPUT_DIR)
    for name in ("log", "recommend_topk", "saved"):
        src = Path(name)
        dst = output_root / name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    output_volume.commit()
    return f"Finished training {model} on {dataset}. Outputs are in Modal volume 'freedom-runs'."


@app.local_entrypoint()
def main(
    model: str = "FREEDOM",
    dataset: str = "hm",
    epochs: int = 0,
    stopping_step: int = 20,
    early_stopping: bool = True,
    early_stopping_min_delta: float = 0.0,
    train_batch_size: int = A100_TRAIN_BATCH_SIZE,
    eval_batch_size: int = A100_EVAL_BATCH_SIZE,
    save_model: bool = True,
):
    call = train_remote.spawn(
        model=model,
        dataset=dataset,
        epochs=epochs,
        stopping_step=stopping_step,
        early_stopping=early_stopping,
        early_stopping_min_delta=early_stopping_min_delta,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        save_model=save_model,
    )
    print(f"Started Modal training call: {call.object_id}")
    print(call.get())
