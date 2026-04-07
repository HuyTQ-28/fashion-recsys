import modal

app = modal.App("hgnn-training")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "pandas", "numpy", "tqdm")
    .pip_install("torch-geometric", "torch-scatter", "torch-sparse")
    .add_local_dir(".", remote_path="/root", ignore=["venv", "__pycache__", ".git"])
)

volume = modal.Volume.from_name("hgnn-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": volume},
    timeout=60*60
)
def train_model():
    import sys
    sys.path.append("/root")

    import os
    os.environ["DATA_DIR"] = "/data"
    
    from hgnn_model.train import train
    train()