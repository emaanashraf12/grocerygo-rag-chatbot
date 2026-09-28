from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class FAQVectorStore:
    """
    Creates and manages the FAISS vector store used
    for semantic FAQ retrieval.
    """

    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vector_store = None

    def create_vector_store(self, documents):
        """
        Convert FAQ documents into embeddings and
        store them in a FAISS vector index.
        """

        if not documents:
            raise ValueError("No documents were provided.")

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )

        return self.vector_store

    def search(self, query, k=3):
        """
        Retrieve the most semantically relevant FAQs.
        """

        if self.vector_store is None:
            raise ValueError(
                "Vector store has not been created yet."
            )

        return self.vector_store.similarity_search(
            query,
            k=k
        )