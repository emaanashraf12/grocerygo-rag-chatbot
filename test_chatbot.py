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


def main():

    print("=" * 70)
    print("GROCERYGO KAGGLE RAG CHATBOT TEST")
    print("=" * 70)

    print(
        "\nLoading Kaggle "
        "customer-support dataset..."
    )

    documents, df = (
        load_faq_data(
            DATASET_PATH
        )
    )

    print(
        f"Processed dataset: "
        f"{len(df)} records."
    )

    print(
        "\nCreating FAISS "
        "vector store..."
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

    print(
        "FAISS vector store "
        "created successfully."
    )

    print(
        "\nStarting GroceryGo chatbot..."
    )

    chatbot = (
        GroceryFAQChatbot(
            vector_store
        )
    )

    print(
        "Chatbot ready!"
    )

    test_questions = [
        "I forgot my password. "
        "How can I recover it?",

        "Where is my order? "
        "Can I track it?",

        "What payment methods "
        "can I use?",

        "How long does delivery take?",

        "I want to request a refund.",

        "Can I change the address "
        "for my order?",

        "I need to speak to "
        "a human support agent.",

        "Who invented the telephone?"
    ]

    for number, question in enumerate(
        test_questions,
        start=1
    ):

        print("\n" + "=" * 70)

        print(
            f"TEST {number}"
        )

        print("=" * 70)

        print(
            "\nUSER:"
        )

        print(
            question
        )

        try:

            result = (
                chatbot.answer(
                    question
                )
            )

            print(
                "\nGROCERYGO:"
            )

            print(
                result["answer"]
            )

            print(
                "\nRETRIEVED "
                "KAGGLE EXAMPLES:"
            )

            if result["sources"]:

                for index, document in enumerate(
                    result["sources"],
                    start=1
                ):

                    print(
                        f"{index}. "
                        f"[{document.metadata.get('intent')}] "
                        f"{document.metadata.get('question')}"
                    )

            else:

                print(
                    "No sources retrieved."
                )

        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                error
            )

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()