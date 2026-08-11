"""Streamlit chat-style demo: query by audio, image, or text; see cross-modal retrieval
results with LLM explanations; answer a clarifying follow-up when species are ambiguous.

Run with: uv run streamlit run birdcall/app/streamlit_app.py
"""

import tempfile
from pathlib import Path

import streamlit as st
import torch

from birdcall.embed.imagebind_encoder import ImageBindEncoder
from birdcall.index.faiss_index import BirdIndex
from birdcall.model.projection_head import ProjectionHead
from birdcall.reasoning.llm_reasoning import ReasoningLayer
from birdcall.species import CONFUSABLE_PAIRS

# Must be the index built WITH --projection-head (see scripts/run_pipeline.py `index`), since
# the query embedding below is also projected through the same head before searching.
INDEX_DIR = "artifacts/index/finetuned"
PROJECTION_HEAD_PATH = "artifacts/projection_head/projection_head.pt"


@st.cache_resource
def load_resources():
    encoder = ImageBindEncoder()
    index = BirdIndex.load(INDEX_DIR)
    head = ProjectionHead(in_dim=1024)
    head.load_state_dict(torch.load(PROJECTION_HEAD_PATH, map_location="cpu"))
    head.eval()
    reasoning = ReasoningLayer(confusable_pairs=CONFUSABLE_PAIRS)
    return encoder, index, head, reasoning


def embed_query(encoder, head, modality: str, payload: str):
    with torch.no_grad():
        raw = encoder.encode_modality(modality, [payload])
        projected = head(raw)
    return projected.cpu().numpy()[0]


def render_results(results, reasoning_result) -> None:
    for r, expl in zip(results, reasoning_result.explanations):
        flag = " :warning: low confidence" if expl["low_confidence"] else ""
        st.markdown(f"**{r['species']}** ({r['modality']}, score={r['score']:.3f}){flag}")
        st.caption(expl["explanation"])
    if reasoning_result.clarifying_question:
        st.info(reasoning_result.clarifying_question)


def _save_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def main():
    st.set_page_config(page_title="Bird Call Cross-Modal Retrieval", page_icon="🐦")
    st.title("🐦 Bird Call Cross-Modal Retrieval")
    st.caption("Query with a bird call, a photo, or a text description — retrieve matches across all three.")

    encoder, index, head, reasoning = load_resources()

    tab_audio, tab_image, tab_text = st.tabs(["Audio", "Image", "Text"])
    query_modality, query_payload, query_description = None, None, None

    with tab_audio:
        audio_file = st.file_uploader("Upload a bird call (wav/mp3)", type=["wav", "mp3"], key="audio")
        if audio_file is not None:
            query_modality, query_description = "audio", "an uploaded bird call recording"
            query_payload = _save_temp(audio_file)

    with tab_image:
        image_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="image")
        if image_file is not None:
            query_modality, query_description = "image", "an uploaded bird photo"
            query_payload = _save_temp(image_file)

    with tab_text:
        text_query = st.text_input("Describe the bird or its call")
        if text_query:
            query_modality, query_description, query_payload = "text", text_query, text_query

    if query_modality and st.button("Search"):
        vec = embed_query(encoder, head, query_modality, query_payload)
        results = index.search(vec, k=5)
        st.session_state.last_query_description = query_description
        st.session_state.last_results = results
        st.session_state.last_reasoning = reasoning.explain(query_description, results)

    if "last_results" in st.session_state:
        render_results(st.session_state.last_results, st.session_state.last_reasoning)
        if st.session_state.last_reasoning.clarifying_question:
            answer = st.text_input("Your answer", key="clarify_input")
            if answer and st.button("Refine"):
                refine_desc = f"{st.session_state.last_query_description}. Additional context: {answer}"
                st.session_state.last_reasoning = reasoning.explain(refine_desc, st.session_state.last_results)
                st.rerun()


if __name__ == "__main__":
    main()
