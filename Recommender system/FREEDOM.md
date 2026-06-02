# Act as a Senior Machine Learning Engineer specializing in Graph Recommender Systems.

# Context:
I need to train, evaluate and qualitatively validate the FREEDOM multimodal recommendation model using the H&M Personalized Fashion Recommendation dataset from Kaggle. To minimize engineering effort, I want to strictly leverage the official FREEDOM GitHub repository (https://github.com/enoche/FREEDOM) and its underlying framework (MMRec), modifying the code only where absolutely necessary.

# Dataset specifics:
- images/ - a folder of images corresponding to each article_id; images are placed in subfolders starting with the first three digits of the article_id; note, not all article_id values have a corresponding image.

- articles.csv: article_id, product_code, prod_name, product_type_no, product_type_name, product_group_name, graphical_appearance_no, graphical_appearance_name, colour_group_code, colour_group_name, perceived_colour_value_id, perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name, department_no, department_name, index_code

- customers.csv: customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code, index_name, index_group_no, index_group_name, section_no, section_name, garment_group_no, garment_group_name, detail_desc

- transactions_train.csv - the training data, consisting of the purchases each customer for each date, as well as additional information. Duplicate rows correspond to multiple purchases of the same item. Fields: t_dat, customer_id, article_id, price, sales_channel_id

# Task: Please provide a comprehensive, step-by-step technical execution plan covering the following deliverables:

1. Optimal Data Sampling Strategy:
Formulate the best sampling strategy for the H&M dataset to construct the graph. Keep in mind the extreme seasonality, fast-fashion data drift, and graph connectivity requirements. Explicitly evaluate why random sampling across different seasons fails, and propose a time-based "sliding window" or continuous recent data approach (e.g., using the last 6-8 weeks).

2. Multimodal Data Preprocessing Pipeline:
Detail how to process the raw H&M data to match the exact input format expected by the FREEDOM codebase.

- How should I extract visual features (e.g., ResNet, FashionCLIP, etc.) for items with images, and handle items with missing images?

- How should I extract textual features (e.g., Sentence-BERT, etc.) from articles.csv?

- Specify the K-core filtering rules to reduce sparsity and clean the user-item bipartite graph.

3. Adapting the GitHub Repository:
Walk me through the exact steps to integrate my preprocessed H&M dataset into the downloaded FREEDOM repository.

- What specific files, folder structures, or config files (e.g., YAML files in the MMRec framework) need to be modified or created?

- Do I need to modify the data loader or the graph construction scripts?

4. Training and Evaluation Execution:
Provide the exact terminal commands to initiate the pre-training of the frozen item-item graph and the subsequent main training loop. Include how to evaluate the model using the standard metrics (Recall@20, NDCG@20) and how to output predictions for H&M's standard MAP@12 metric.

5. Qualitative Validation (Sanity Checks):

- Visual Nearest Neighbors: A way to extract the top-K visually similar items in the learned latent space for a given anchor item to see if the graph learned aesthetic relationships.

- t-SNE/UMAP Visualization: Instructions on how to project the final Item Embeddings ($h_i$) into 2D, colored by product_group_name from articles.csv, to verify proper clustering.