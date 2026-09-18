import os
import pickle

import faiss
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hospital Knowledge Assistant",
    page_icon="🏥",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_DIR = "faiss_index"

FAISS_INDEX_PATH = os.path.join(
    INDEX_DIR,
    "index.faiss"
)

METADATA_PATH = os.path.join(
    INDEX_DIR,
    "metadata.pkl"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 5


# ============================================================
# PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main {
        max-width: 900px;
        margin: auto;
    }

    .source-box {
        padding: 10px 14px;
        border-radius: 8px;
        background-color: #f5f7fa;
        border: 1px solid #e1e5ea;
        margin-bottom: 8px;
    }

    .source-title {
        font-weight: 600;
        margin-bottom: 3px;
    }

    .source-details {
        font-size: 0.88rem;
        color: #555;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🏥 Hospital Knowledge Assistant")

st.caption(
    "Ask questions about the hospital policies and documents "
    "in the knowledge base."
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():

    if not os.path.exists(FAISS_INDEX_PATH):

        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}"
        )

    index = faiss.read_index(
        FAISS_INDEX_PATH
    )

    return index


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_resource
def load_metadata():

    if not os.path.exists(METADATA_PATH):

        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "rb"
    ) as f:

        metadata = pickle.load(f)

    return metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

try:

    index = load_faiss_index()

    metadata = load_metadata()

    embedding_model = load_embedding_model()

except Exception as e:

    st.error(
        "The knowledge base could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# GROQ CLIENT
# ============================================================

try:

    groq_api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    st.error(
        "GROQ_API_KEY was not found in Streamlit Secrets."
    )

    st.info(
        "Add your Groq API key as GROQ_API_KEY "
        "in your Streamlit app Secrets."
    )

    st.stop()


client = Groq(
    api_key=groq_api_key
)


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_chunks(question, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")


    scores, indices = index.search(
        query_embedding,
        top_k
    )


    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(metadata):
            continue

        result = metadata[idx].copy()

        result["similarity"] = float(score)

        results.append(result)


    return results


# ============================================================
# GENERATE ANSWER WITH GROQ
# ============================================================

def generate_answer(question, retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
Department: {chunk.get("department", "Unknown")}
Document: {chunk.get("source_file", "Unknown")}
Page: {chunk.get("page", "Unknown")}

CONTENT:
{chunk.get("text", "")}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    system_prompt = """
You are a hospital policy knowledge assistant.

Your job is to answer questions using ONLY the
provided hospital document context.

Rules:

1. Use the supplied context as the source of truth.
2. Do not invent hospital policies, procedures,
   requirements, numbers, dates, or rules.
3. If the answer is not supported by the supplied
   documents, clearly say that the information was
   not found in the hospital knowledge base.
4. Give a concise, useful answer.
5. When appropriate, mention the relevant department.
6. Do not cite information that is not present in
   the supplied context.
7. Do not use outside knowledge to fill missing
   hospital policy information.
"""


    user_prompt = f"""
Hospital document context:

{context}

User question:

{question}

Answer the question based only on the hospital
document context above.
"""


    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2,

        max_completion_tokens=1500,

        include_reasoning=False
    )


    return response.choices[0].message.content


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            st.markdown(
                "**Sources**"
            )

            for source in message["sources"]:

                st.markdown(
                    f"""
                    <div class="source-box">

                    <div class="source-title">
                    📄 {source["source_file"]}
                    </div>

                    <div class="source-details">
                    Department: {source["department"]}
                    &nbsp; | &nbsp;
                    Page: {source["page"]}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about hospital policies..."
)


if question:

    # --------------------------------------------------------
    # SHOW USER QUESTION
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # RETRIEVE DOCUMENT CHUNKS
    # --------------------------------------------------------

    with st.spinner(
        "Searching the hospital knowledge base..."
    ):

        retrieved_chunks = retrieve_chunks(
            question,
            TOP_K
        )


    if not retrieved_chunks:

        answer = (
            "I could not find relevant information "
            "in the hospital knowledge base."
        )

        sources = []

    else:

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            try:

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            except Exception as e:

                st.error(
                    "The Groq request failed."
                )

                st.exception(e)

                st.stop()


        # ----------------------------------------------------
        # REMOVE DUPLICATE SOURCES
        # ----------------------------------------------------

        sources = []

        seen = set()

        for chunk in retrieved_chunks:

            source_key = (
                chunk.get("source_path"),
                chunk.get("page")
            )

            if source_key not in seen:

                seen.add(
                    source_key
                )

                sources.append(
                    {
                        "department":
                            chunk.get(
                                "department",
                                "Unknown"
                            ),

                        "source_file":
                            chunk.get(
                                "source_file",
                                "Unknown"
                            ),

                        "source_path":
                            chunk.get(
                                "source_path",
                                ""
                            ),

                        "page":
                            chunk.get(
                                "page",
                                "Unknown"
                            )
                    }
                )


    # --------------------------------------------------------
    # SHOW ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)

        if sources:

            st.markdown(
                "**Sources**"
            )

            for source in sources:

                st.markdown(
                    f"""
                    <div class="source-box">

                    <div class="source-title">
                    📄 {source["source_file"]}
                    </div>

                    <div class="source-details">
                    Department: {source["department"]}
                    &nbsp; | &nbsp;
                    Page: {source["page"]}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )
