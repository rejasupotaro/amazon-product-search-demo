import pickle

import numpy as np
import pandas as pd
from amazon_product_search_dense_retrieval.encoders import BERTEncoder
from invoke import task


@task
def encode(c):
    products_df = pd.read_csv("data/products_small_jp.csv.zip")
    products = products_df.to_dict("records")

    product_ids = [product["product_id"] for product in products]
    id_filepath = "data/product_ids.pkl"
    with open(id_filepath, "wb") as file:
        pickle.dump(product_ids, file)
        print(f"{id_filepath} was saved.")

    product_titles = [product["product_title"] for product in products]
    for rep_mode in ["cls", "mean", "max"]:
        encoder = BERTEncoder(bert_model_name="ku-nlp/deberta-v2-base-japanese", rep_mode=rep_mode)
        title_embs = encoder.encode(product_titles)
        emb_filepath = f"data/title_embs_{rep_mode}.npy"
        with open(emb_filepath, "wb") as file:
            np.save(file, title_embs)
            print(f"{emb_filepath} was saved.")


@task
def app(c):
    c.run("poetry run streamlit run src/Sparse_Retrieval.py")


@task
def export_dependencies(c):
    c.run("poetry export --without-hashes --format=requirements.txt > requirements.txt")
