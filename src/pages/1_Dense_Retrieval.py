import pickle
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from amazon_product_search_dense_retrieval.encoders import BERTEncoder
from amazon_product_search_dense_retrieval.retrievers import SingleVectorRetriever


@st.cache_resource
def load_encoder(rep_mode: str) -> BERTEncoder:
    return BERTEncoder(bert_model_name="ku-nlp/deberta-v2-base-japanese", rep_mode=rep_mode)


@st.cache_data
def load_product_dict() -> dict[str, Any]:
    products_df = pd.read_csv("data/products_small_jp.csv.zip")
    product_dict: dict[str, Any] = {}
    for row in products_df.to_dict("records"):
        product_dict[row["product_id"]] = row
    return product_dict


def main():
    st.write("## Dense Retrieval")

    st.write("### Input")
    rep_mode = st.selectbox("rep_mode", options=["cls", "mean", "max"], index=0)
    query = st.text_input("query")
    if not query:
        return

    with open("data/product_ids.pkl", "rb") as file:
        product_ids = pickle.load(file)
    with open(f"data/title_embs_{rep_mode}.npy", "rb") as file:
        title_embs = np.load(file)
    retriever = SingleVectorRetriever(
        dim=title_embs.shape[1],
        doc_ids=product_ids,
        doc_embs=title_embs,
    )
    product_dict = load_product_dict()
    encoder = load_encoder(rep_mode)

    st.write("### Results")
    query_vec = encoder.encode([query])[0]
    retrieved = retriever.retrieve(query=query_vec, top_k=10)
    rows = []
    for product_id, score in zip(*retrieved):
        if product_id not in product_dict:
            continue
        product = product_dict[product_id]
        rows.append(
            {
                "score": score,
                **product,
            }
        )
    scores_df = pd.DataFrame(rows)
    scores_df = scores_df.sort_values("score", ascending=False)
    st.write(scores_df)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Amazon Product Search",
        page_icon="🛍️",
        layout="wide",
    )
    main()
