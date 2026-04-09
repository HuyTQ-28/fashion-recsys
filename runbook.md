# Real-Time Personalized Fashion Recommender: Execution Runbook

This guide outlines every exact step you need to run to spin up your architecture completely from scratch.

> [!IMPORTANT]
> Make sure your virtual environment is active before starting!
> `source venv/bin/activate`

---

## Phase 1: Data Preparation

1. **Download Kaggle Data**: Download the *H&M Personalized Fashion Recommendations* dataset from Kaggle.
2. Put the `transactions_train.csv` and `articles.csv` into a folder named `data/` in the root of your project. Also place the `images/` directory inside `data/` if you want local thumbnails.
3. **Generate Graph Interaction Data**: The Kaggle dataset only has purchases. We need to simulate clicks and carts so the HGNN graph has a heterogeneous structure.
   ```bash
   python src/hgnn_model/generate_fake_behavior.py
   ```
   *(This generates `data/fake_behavior.csv` which takes the last 2M rows and assigns random events).*

## Phase 2: Upload Files to Modal

Modal acts as your serverless backend for training and deployment. We need to upload `clip_embeddings.pt` and the newly generated `fake_behavior.csv` into the persistent Modal volume.

```bash
# Create the checkpoints volume to host data between Modal runs
modal volume create checkpoints

# Upload the files
modal volume put checkpoints data/fake_behavior.csv /fake_behavior.csv
# Assuming you've already uploaded clip_embeddings, but if you haven't:
modal volume put checkpoints data/clip_embeddings.pt /clip_embeddings.pt
```

## Phase 3: Model Training (Modal GPUs)

**1. Train HGNN Teacher Model**  
The graph neural network builds item-item relations.

```bash
modal run src/hgnn_model/run_hgnn.py
```
*Expected Output: This spins up an A100/T4 instance, consumes `/checkpoints/fake_behavior.csv`, runs graph contrastive learning, and drops `/checkpoints/article_embeddings_hgnn3.pt` into Modal.*

**2. Train Student MLP Model**  
Distills the 64-dim HGNN knowledge to project 512-dim CLIP vectors.

```bash
modal run src/MLP_student/run_student.py
```
*Expected Output: A new file is saved directly into the modal volume at `/checkpoints/student_mlp_full.pt`.*

## Phase 4: Vector Indexing (Weaviate Cloud)

To populate Weaviate, you must pull down the compiled Graph embeddings from Modal to your local machine so the ingest script can batch-load Weaviate.

1. **Download the models locally**:
   ```bash
   modal volume get checkpoints /article_embeddings_hgnn3.pt data/article_embeddings_hgnn3.pt
   modal volume get checkpoints /student_mlp_full.pt data/student_mlp_full.pt
   ```
2. **Setup your `.env`**: Make sure your `.env` file contains Weaviate cloud credentials (`WEAVIATE_URL` and `WEAVIATE_API_KEY`).
3. **Provision the collections**:
   ```bash
   python src/search/weaviate_setup.py
   ```
4. **Push data to Weaviate**:
   ```bash
   python src/search/ingest_products.py
   ```

## Phase 5: Deploy Backend API

Deploy the API server that processes the Triplet Loss and Hybrid Searches in real-time.

```bash
modal deploy src/modal_app/app.py
```
*Warning: Once complete, Modal will output your unique runtime URL. Save it.*

## Phase 6: Run Streamlit

Export the Modal endpoint to your environment so the frontend knows where the serverless backend is sitting.

```bash
export MODAL_API_BASE_URL="https://[YOUR-MODAL-APP-URL]"
streamlit run app/streamlit_app.py
```
