import pandas as pd
import streamlit as st
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


class Encoder:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, text: str) -> Tensor:
        with torch.no_grad():
            tokens = self.tokenizer(
                [text],
                add_special_tokens=True,
                padding="longest",
                truncation="longest_first",
                return_attention_mask=True,
                return_tensors="pt",
            )

            attention_mask = tokens["attention_mask"]
            vecs = self.model(input_ids=tokens["input_ids"], attention_mask=attention_mask).last_hidden_state
            vec = (vecs * attention_mask.unsqueeze(-1)).mean(dim=1)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec


encoder = Encoder("ku-nlp/deberta-v2-base-japanese")


@st.cache_data
def load_products() -> pd.DataFrame:
    return pd.read_csv("data/products_small_jp.csv.zip")


def main():
    st.write("## Amazon Product Search")
    products_df = load_products()
    products = products_df.head(10).to_dict("records")

    st.write("### Input")
    query = st.text_input("query")
    if not query:
        return

    st.write("### Results")
    query_vec = encoder.encode(query)
    rows = []
    for product in products:
        title = product["product_title"]
        product_vec = encoder.encode(title)
        score = (query_vec * product_vec).sum(dim=1).numpy()
        rows.append(
            {
                "title": title,
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
