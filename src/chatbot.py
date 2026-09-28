import os
from collections import Counter

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


FALLBACK_RESPONSE = (
    "I don't have enough information in the GroceryGo "
    "knowledge base to answer that question. "
    "Please contact customer support."
)


class GroceryFAQChatbot:
    """
    Retrieval-Augmented Generation chatbot for GroceryGo.

    Customer utterances from the Kaggle Bitext customer-support
    dataset are embedded and stored in FAISS.

    At query time, similar customer utterances are retrieved.
    Their intent labels and approved GroceryGo responses are
    provided to Gemini as grounded context.
    """

    def __init__(self, vector_store):

        self.vector_store = vector_store

        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY was not found. "
                "Please add it to the .env file."
            )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key
        )

        self.prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are GroceryGo Assistant, a customer-support
chatbot for a fictional online grocery delivery service.

You will receive approved GroceryGo support information
retrieved from the knowledge base.

Follow these rules strictly:

1. Answer using ONLY the approved information supplied
   in the context.

2. Do not use outside knowledge.

3. Do not invent policies, prices, delivery times,
   payment methods, refund guarantees, discounts,
   or company procedures.

4. The retrieved context contains customer utterances,
   intent labels, and approved GroceryGo responses.
   Use the approved responses as the factual source
   for your answer.

5. If the user's question is unrelated to GroceryGo,
   online ordering, delivery, payment, refunds,
   accounts, or customer support, respond exactly:

   I don't have enough information in the GroceryGo
   knowledge base to answer that question. Please
   contact customer support.

6. If the supplied context does not contain enough
   information to answer the question, use the same
   insufficient-information response.

7. Keep valid answers concise, clear, and helpful.

8. Do not mention Kaggle, FAISS, embeddings, LangChain,
   retrieved documents, prompts, intents, or other
   internal implementation details to the customer.
                        """
                    ),
                    (
                        "human",
                        """
KNOWLEDGE BASE CONTEXT:

{context}

USER QUESTION:

{question}

Answer the user's question using only the approved
information in the context.
                        """
                    )
                ]
            )
        )

    def retrieve(
        self,
        question,
        k=5
    ):
        """
        Retrieve semantically similar customer-support
        utterances from FAISS.
        """

        if (
            not question
            or not question.strip()
        ):
            return []

        return (
            self.vector_store
            .similarity_search(
                question,
                k=k
            )
        )

    @staticmethod
    def extract_text(response):
        """
        Extract visible text from Gemini responses.
        """

        content = response.content

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):

            text_parts = []

            for block in content:

                if isinstance(
                    block,
                    dict
                ):

                    if (
                        block.get("type")
                        == "text"
                    ):

                        text = block.get(
                            "text",
                            ""
                        )

                        if text:
                            text_parts.append(
                                text
                            )

                elif isinstance(
                    block,
                    str
                ):
                    text_parts.append(
                        block
                    )

            return "\n".join(
                text_parts
            ).strip()

        return str(content).strip()

    @staticmethod
    def select_context_documents(
        documents
    ):
        """
        Determine the dominant intent among the retrieved
        examples and use documents from that intent as
        grounded context.

        This reduces the chance of mixing unrelated
        responses from neighboring intents.
        """

        if not documents:
            return []

        intents = [
            document.metadata.get(
                "intent"
            )
            for document in documents
            if document.metadata.get(
                "intent"
            )
        ]

        if not intents:
            return documents[:3]

        dominant_intent = (
            Counter(intents)
            .most_common(1)[0][0]
        )

        selected = [
            document
            for document in documents
            if document.metadata.get(
                "intent"
            ) == dominant_intent
        ]

        return selected[:3]

    def answer(self, question):
        """
        Complete RAG pipeline.
        """

        if (
            not question
            or not question.strip()
        ):
            return {
                "answer":
                    "Please enter a question "
                    "about GroceryGo.",
                "sources": []
            }

        retrieved_documents = (
            self.retrieve(
                question=question,
                k=5
            )
        )

        if not retrieved_documents:
            return {
                "answer":
                    FALLBACK_RESPONSE,
                "sources": []
            }

        context_documents = (
            self.select_context_documents(
                retrieved_documents
            )
        )

        if not context_documents:
            return {
                "answer":
                    FALLBACK_RESPONSE,
                "sources": []
            }

        context_parts = []

        for document in (
            context_documents
        ):

            context_parts.append(
                document.page_content
            )

        context = (
            "\n\n---\n\n".join(
                context_parts
            )
        )

        chain = (
            self.prompt
            | self.llm
        )

        response = chain.invoke(
            {
                "context": context,
                "question":
                    question.strip()
            }
        )

        answer_text = (
            self.extract_text(
                response
            )
        )

        if not answer_text:
            answer_text = (
                FALLBACK_RESPONSE
            )

        return {
            "answer": answer_text,
            "sources":
                context_documents
        }