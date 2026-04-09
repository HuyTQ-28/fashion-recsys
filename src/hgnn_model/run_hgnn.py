import modal

app = modal.App("hgnn-training")

TORCH_VERSION = "2.2.1"
CUDA_VERSION = "cu121"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        f"torch=={TORCH_VERSION}+{CUDA_VERSION}", 
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("pandas", "numpy<2.0.0", "tqdm")
    .pip_install("torch-geometric")
    .run_commands(
        f"pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html"
    )
    
    .add_local_dir("src", remote_path="/root", ignore=["venv", "__pycache__", ".git"])
)

volume = modal.Volume.from_name("checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=60*60*4
)
def train_model():
    import sys
    sys.path.append("/root")

    import os
    os.environ["DATA_DIR"] = "/checkpoints"
    
    from hgnn_model.train import train
    train()

@app.local_entrypoint()
def main():
    train_model.remote()