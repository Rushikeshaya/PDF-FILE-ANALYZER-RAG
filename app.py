
import hashlib
from pathlib import Path

import streamlit as st
import pymupdf
from sentence_transformers import SentenceTransformer

from db import (
    initialize_database,
    document_exists,
    save_document,
    hybrid_search,
)

MODEL_PATH = Path(__file__).parent / "model" / "all-MiniLM-L6-v2"

st.set_page_config(
    page_title="Offline PDF RAG",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_model():
    return SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True,
    )


def extract_pdf(pdf_bytes):
    chunks = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            # Split text into overlapping chunks.
            chunk_size = 1000
            overlap = 150
            start = 0

            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({
                        "page_number": page_number,
                        "text": chunk_text,
                    })

                if end == len(text):
                    break

                start = end - overlap

    return chunks


st.title("📚 Offline PDF RAG")
st.caption(
    "Search your PDF documents locally using semantic and keyword search."
)

st.info(
    "This application retrieves relevant passages from your PDFs. "
    "It does not use a generative LLM to write answers."
)

try:
    initialize_database()
    model = load_model()
except Exception as error:
    st.error(f"Initialization failed: {error}")
    st.stop()


st.header("1. Upload a PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    if st.button("Process PDF", type="primary"):
        pdf_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()

        if document_exists(file_hash):
            st.warning("This PDF has already been processed.")
        else:
            try:
                with st.spinner("Extracting PDF text..."):
                    chunks = extract_pdf(pdf_bytes)

                if not chunks:
                    st.error(
                        "No selectable text was found. "
                        "Scanned PDFs may require OCR."
                    )
                else:
                    progress = st.progress(0)
                    batch_size = 32
                    all_embeddings = []

                    with st.spinner(
                        f"Generating embeddings for {len(chunks)} chunks..."
                    ):
                        for start in range(0, len(chunks), batch_size):
                            batch = chunks[start:start + batch_size]

                            embeddings = model.encode(
                                [chunk["text"] for chunk in batch],
                                normalize_embeddings=True,
                                show_progress_bar=False,
                            )

                            all_embeddings.extend(embeddings)
                            progress.progress(
                                min(
                                    (start + len(batch)) / len(chunks),
                                    1.0,
                                )
                            )

                    saved = save_document(
                        uploaded_file.name,
                        file_hash,
                        chunks,
                        all_embeddings,
                    )

                    if saved:
                        st.success(
                            f"Processed {uploaded_file.name} successfully!"
                        )
                        st.write(f"Pages with text: {len(set(
                            chunk['page_number'] for chunk in chunks
                        ))}")
                        st.write(f"Text chunks saved: {len(chunks)}")
                    else:
                        st.warning("This PDF already exists in the database.")

            except Exception as error:
                st.error(f"PDF processing failed: {error}")


st.divider()
st.header("2. Ask a question")

question = st.text_input(
    "What would you like to find in your PDFs?",
    placeholder="Example: What is the main objective of this document?",
)

result_count = st.slider(
    "Number of passages to retrieve",
    min_value=1,
    max_value=10,
    value=5,
)

if st.button("Search documents", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching your documents..."):
                query_embedding = model.encode(
                    question,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                results = hybrid_search(
                    question,
                    query_embedding,
                    limit=result_count,
                )

            if not results:
                st.warning("No relevant passages were found.")
            else:
                st.subheader("Retrieved passages")

                for index, result in enumerate(results, start=1):
                    with st.expander(
                        f"{index}. {result['filename']} — "
                        f"Page {result['page_number']}",
                        expanded=(index == 1),
                    ):
                        st.write(result["text"])
                        st.caption(
                            "Hybrid retrieval score: "
                            f"{result['score']:.5f}"
                        )

        except Exception as error:
            st.error(f"Search failed: {error}")


st.divider()
st.caption(
    "Local embeddings • PostgreSQL • pgvector • "
    "Full-text search • Reciprocal Rank Fusion"
)
