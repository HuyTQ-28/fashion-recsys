import modal

app = modal.App("mlp-distill")

# ===== Cấu hình môi trường =====
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_dir(".", remote_path="/app")  # Thêm mã nguồn
)

volume = modal.Volume.from_name("recsys-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume}
)
def train_model():
    import sys
    sys.path.append("/app")  # Cho phép import module local

    import torch
    from torch.utils.data import DataLoader

    # Import mô hình và dataset
    from MLP_student.model import StudentMLP
    from MLP_student.build_dataset import DistillDataset
    from MLP_student.train import train_student  # Hàm train tối ưu

    # ===== LOAD DATA =====
    data = torch.load("/data/article_embeddings_hgnn3.pt")
    clip_dict = torch.load("/data/clip_embeddings.pt")

    id2idx = data["id2idx"]
    hgnn_embeddings = data["embeddings"]

    dataset = DistillDataset(id2idx, hgnn_embeddings, clip_dict)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # ===== KHỞI TẠO MÔ HÌNH =====
    clip_sample = dataset[0][0]
    model = StudentMLP(
        in_dim=clip_sample.shape[0],
        out_dim=hgnn_embeddings.shape[1]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== TRAIN MÔ HÌNH SỬ DỤNG HÀM train_student =====
    trained_model = train_student(
        model,
        loader,
        epochs=130,
        lr=1e-3,
        patience=3,
        device=device,
        use_wandb=False  # Có thể bật nếu muốn
    )

    # ===== LƯU MÔ HÌNH =====
    torch.save(trained_model.state_dict(), "/data/mlp_student.pt")
    print("Saved trained MLP model to /data/mlp_student.pt")