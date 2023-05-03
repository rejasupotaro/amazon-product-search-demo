import pickle

import numpy as np
import pandas as pd
from amazon_product_search_dense_retrieval.encoders import BERTEncoder
from invoke import task


@task
def encode(c):
    products_df = pd.read_csv("data/products_small_jp.csv.zip").head(100)
    products = products_df.to_dict("records")
    encoder = BERTEncoder(bert_model_name="ku-nlp/deberta-v2-base-japanese")
    product_ids = [product["product_id"] for product in products]
    product_titles = [product["product_title"] for product in products]
    title_embs = encoder.encode(product_titles)

    with open("data/product_ids.pkl", "wb") as file:
        pickle.dump(product_ids, file)
    with open("data/title_embs.npy", "wb") as file:
        np.save(file, title_embs)


@task
def app(c):
    c.run("poetry run streamlit run src/Sparse_Retrieval.py")


@task
def export_dependencies(c):
    c.run("poetry export --without-hashes --format=requirements.txt > requirements.txt")


@task
def format(c):
    """Run formatters (isort and black)."""
    print("Running isort...")
    c.run("poetry run isort .")

    print("Running black...")
    c.run("poetry run black .")
    print("Done")


@task
def lint(c):
    """Run linters (isort, black, flake8, and mypy)."""
    print("Running isort...")
    c.run("poetry run isort . --check")

    print("Running black...")
    c.run("poetry run black . --check")

    print("Running flake8...")
    c.run("poetry run pflake8 src tests")

    print("Running mypy...")
    c.run("poetry run mypy src")
    print("Done")
