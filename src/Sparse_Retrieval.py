import pandas as pd
import streamlit as st


@st.cache_data
def load_products() -> pd.DataFrame:
    return pd.read_csv("data/products_small_jp.csv.zip")


def main():
    st.write("## Amazon Product Search")
    products_df = load_products()

    st.write("### Input")
    query = st.text_input("query")
    query_tokens = query.split()

    st.write("### Results")
    product_dicts = products_df.to_records("records")
    top_k = 10
    fields = ["product_title", "product_brand", "product_color", "product_bullet_point"]
    columns = st.columns(len(fields))
    for column, field in zip(columns, fields):
        with column:
            st.write(f"#### {field}")
            products = [
                product
                for product in product_dicts
                if any(query_token in str(product[field]) for query_token in query_tokens)
            ]
            for i, product in enumerate(products[:top_k]):
                rank = i + 1
                product_title = product["product_title"]
                text = product[field]
                score = 1.0

                st.markdown(f"{rank}. {product_title} (score: {score})")
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
