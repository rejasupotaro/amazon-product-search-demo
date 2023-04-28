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
    products = [product for product in products_df.to_dict("records") if query in product["product_title"]]

    st.write("### Results")
    top_k = 10
    for i, product in enumerate(products[:top_k]):
        rank = i + 1
        product_title = product["product_title"]
        product_description = product["product_description"]
        score = 1.0

        st.markdown(f"{rank}. {product_title} (score: {score})")
        if not pd.isnull(product_description):
            st.markdown(product_description, unsafe_allow_html=True)
        st.markdown("----")


if __name__ == "__main__":
    main()
