from collections import defaultdict
from typing import DefaultDict

import pandas as pd
import streamlit as st


@st.cache_data
def load_products() -> pd.DataFrame:
    return pd.read_csv("data/products_small_jp.csv.zip")


def main():
    st.write("## Sparse Retrieval")
    products_df = load_products()
    fields = ["product_title", "product_brand", "product_color", "product_bullet_point"]
    default_weights = [1.0, 0.6, 0.4, 0.2]

    st.write("### Input")
    with st.form("input"):
        query = st.text_input("query")
        query_tokens = query.split()

        weight_cols = st.columns(len(fields))
        weights = []
        for i, (field, default_weight) in enumerate(zip(fields, default_weights, strict=True)):
            with weight_cols[i]:
                field_name = " ".join(field.split("_")[1:])
                weight = st.number_input(f"{field_name} weight", value=default_weight)
                weights.append(weight)

        aggregation = st.checkbox("aggregation", value=True)

        submitted = st.form_submit_button("search")
        if not submitted:
            return

    st.write("### Results")
    product_dicts = products_df.to_records("records")
    top_k = 10
    if aggregation:
        candidates: DefaultDict[str, list[float]] = defaultdict(list)
        for i, field in enumerate(fields):
            products = [
                product
                for product in product_dicts
                if any(query_token in str(product[field]) for query_token in query_tokens)
            ]
            for product in products[:top_k]:
                product_id = product["product_id"]
                product_title = product["product_title"]
                score = 1.0 * weights[i]
                candidates[(product_id, product_title)].append(score)

        sorted_candidates = sorted(candidates.items(), key=lambda id_and_score: sum(id_and_score[1]), reverse=True)
        for (product_id, product_title), scores in sorted_candidates[:top_k]:
            score = sum(scores)
            score_text = " + ".join([str(s) for s in scores])
            st.markdown(f"ID: {product_id}, score: {score} ({score_text})")
            st.markdown(product_title)
            st.markdown("----")
    else:
        columns = st.columns(len(fields))
        for i, (column, field) in enumerate(zip(columns, fields, strict=True)):
            with column:
                field_name = " ".join(field.split("_")[1:])
                st.write(f"#### {field_name}")
                products = [
                    product
                    for product in product_dicts
                    if any(query_token in str(product[field]) for query_token in query_tokens)
                ]
                for product in products[:top_k]:
                    product_id = product["product_id"]
                    score = 1.0 * weights[i]
                    st.markdown(f"ID: {product_id}, score: {score}")

                    text = product[field]
                    if not pd.isnull(text):
                        st.markdown(text, unsafe_allow_html=True)
                    st.markdown("----")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Amazon Product Search",
        page_icon="🛍️",
        layout="wide",
    )
    main()
