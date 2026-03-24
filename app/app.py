import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭", layout="centered")

st.title("Sentiment Analyzer")
st.caption("Powered by DistilBERT fine-tuned on SST-2")

text = st.text_area("Inser some text", placeholder="This movie was absolutely amazing!", height=120)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Write something!")
    else:
        with st.spinner("Loading..."):
            try:
                res = requests.post(f"{API_URL}/predict", json={"text": text})
                data = res.json()

                label = data["label"]
                score = data["score"]

                if label == "POSITIVE":
                    st.success(f":) Positive! — confidence: {score:.1%}")
                else:
                    st.error(f"): Negative! — confidence: {score:.1%}")

                st.progress(score)

            except Exception as e:
                st.error(f"Error in API call {e}")