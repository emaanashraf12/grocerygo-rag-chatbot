import streamlit as st

from src.data_loader import (
    load_faq_data
)

from src.vector_store import (
    FAQVectorStore
)

from src.chatbot import (
    GroceryFAQChatbot
)


DATASET_PATH = (
    "bitext_customer_support.csv"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GroceryGo FAQ Chatbot",
    page_icon="🛒",
    layout="centered"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(128, 128, 128, 0.10);
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD CHATBOT
# ============================================================

@st.cache_resource
def initialize_chatbot():
    """
    Load and preprocess the Kaggle Bitext dataset,
    create embeddings, build the FAISS vector store,
    and initialize the GroceryGo chatbot.
    """

    documents, dataframe = (
        load_faq_data(
            DATASET_PATH
        )
    )

    vector_manager = (
        FAQVectorStore()
    )

    vector_store = (
        vector_manager
        .create_vector_store(
            documents
        )
    )

    chatbot = (
        GroceryFAQChatbot(
            vector_store
        )
    )

    return (
        chatbot,
        dataframe
    )


# ============================================================
# INITIALIZE APPLICATION
# ============================================================

try:

    chatbot, dataset = (
        initialize_chatbot()
    )

except Exception as error:

    st.error(
        "The GroceryGo chatbot "
        "could not be initialized."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🛒 GroceryGo'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-Powered Grocery Delivery
        Customer Support Assistant
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
        Ask about orders, delivery, payments,
        refunds, accounts, or customer support.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "About GroceryGo"
    )

    st.write(
        "GroceryGo is a Retrieval-Augmented "
        "Generation (RAG) customer-support "
        "chatbot for a fictional online "
        "grocery delivery service."
    )

    st.divider()

    st.subheader(
        "Knowledge Base"
    )

    st.metric(
        "Processed Records",
        len(dataset)
    )

    st.metric(
        "Support Intents",
        dataset[
            "intent"
        ].nunique()
    )

    st.metric(
        "Categories",
        dataset[
            "display_category"
        ].nunique()
    )

    st.caption(
        "Source dataset: Bitext customer-support "
        "dataset obtained from Kaggle."
    )

    st.divider()

    st.subheader(
        "Support Categories"
    )

    categories = sorted(
        dataset[
            "display_category"
        ].unique()
    )

    for category in categories:

        st.write(
            f"• {category}"
        )

    st.divider()

    st.subheader(
        "Technology"
    )

    st.write(
        """
        • Python  
        • Pandas  
        • LangChain  
        • Hugging Face Embeddings  
        • FAISS  
        • Google Gemini  
        • Streamlit
        """
    )

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm the GroceryGo "
                "Customer Support Assistant. "
                "How can I help you today?"
            ),
            "sources": []
        }
    ]


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
            and message.get(
                "sources"
            )
        ):

            with st.expander(
                "View retrieved "
                "support examples"
            ):

                for index, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.markdown(
                        f"**{index}. "
                        f"{source['question']}**"
                    )

                    st.caption(
                        f"Category: "
                        f"{source['category']}"
                    )

                    if index < len(
                        message["sources"]
                    ):
                        st.divider()


# ============================================================
# CHAT INPUT
# ============================================================

user_question = (
    st.chat_input(
        "Ask a GroceryGo question..."
    )
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
            "sources": []
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_question
        )

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the GroceryGo "
            "knowledge base..."
        ):

            try:

                result = (
                    chatbot.answer(
                        user_question
                    )
                )

                answer = (
                    result["answer"]
                )

                source_data = []

                for document in (
                    result["sources"]
                ):

                    source_data.append(
                        {
                            "question":
                                document.metadata.get(
                                    "question",
                                    "Unknown"
                                ),

                            "category":
                                document.metadata.get(
                                    "category",
                                    "Unknown"
                                ),

                            "intent":
                                document.metadata.get(
                                    "intent",
                                    "Unknown"
                                )
                        }
                    )

            except Exception as error:

                answer = (
                    "Sorry, I encountered an "
                    "error while processing your "
                    "question. Please try again."
                )

                source_data = []

                st.error(
                    f"Technical error: "
                    f"{error}"
                )

        st.markdown(
            answer
        )

        if source_data:

            with st.expander(
                "View retrieved "
                "support examples"
            ):

                for index, source in enumerate(
                    source_data,
                    start=1
                ):

                    st.markdown(
                        f"**{index}. "
                        f"{source['question']}**"
                    )

                    st.caption(
                        f"Category: "
                        f"{source['category']}"
                    )

                    if index < len(
                        source_data
                    ):

                        st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_data
        }
    )