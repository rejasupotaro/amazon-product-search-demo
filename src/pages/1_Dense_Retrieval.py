import pandas as pd
import streamlit as st
from amazon_product_search_dense_retrieval.encoders import BERTEncoder
from amazon_product_search_dense_retrieval.retriever import Retriever


encoder = BERTEncoder(bert_model_name="ku-nlp/deberta-v2-base-japanese")


@st.cache_data
def load_products() -> pd.DataFrame:
    return pd.read_csv("data/products_small_jp.csv.zip")


def main():
    st.write("## Amazon Product Search")
    products_df = load_products()
    products = products_df.head(5).to_dict("records")
    product_ids = [p["product_id"] for p in products]
    product_titles = [p["product_title"] for p in products]
    product_id_to_title = {product_id: product_title for product_id, product_title in zip(product_ids, product_titles)}
    product_embs = encoder.encode(product_titles)
    retriever = Retriever(
        dim=product_embs.shape[1],
        doc_ids=product_ids,
        doc_embs_list=[product_embs],
        weights=[1],
    )

    st.write("### Input")
    query = st.text_input("query")
    if not query:
        return

    st.write("### Results")
    query_vec = encoder.encode([query])[0]
    retrieved = retriever.retrieve(query=query_vec, top_k=5)
    rows = []
    for product_id, score in zip(*retrieved):
        rows.append(
            {
                "product_id": product_id,
                "product_title": product_id_to_title[product_id],
                "score": score,
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
