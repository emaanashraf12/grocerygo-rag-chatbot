from src.data_loader import (
    load_faq_data
)

from src.vector_store import (
    FAQVectorStore
)


DATASET_PATH = (
    "bitext_customer_support.csv"
)


def main():

    print("=" * 70)
    print("GROCERYGO SEMANTIC RETRIEVAL TEST")
    print("=" * 70)

    print(
        "\nLoading and preprocessing "
        "Kaggle dataset..."
    )

    documents, df = (
        load_faq_data(
            DATASET_PATH
        )
    )

    print(
        f"Processed records: "
        f"{len(df)}"
    )

    print(
        "\nCreating embeddings "
        "and FAISS vector store..."
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
        "Vector store created successfully."
    )

    test_cases = [
        {
            "question":
                "I forgot my password. "
                "How can I get back into "
                "my account?",

            "expected_intent":
                "recover_password"
        },

        {
            "question":
                "Where is my order? "
                "I want to track it.",

            "expected_intent":
                "track_order"
        },

        {
            "question":
                "What payment options "
                "can I use?",

            "expected_intent":
                "check_payment_methods"
        },

        {
            "question":
                "I want my money back. "
                "How do I request a refund?",

            "expected_intent":
                "get_refund"
        },

        {
            "question":
                "How long will delivery "
                "take?",

            "expected_intent":
                "delivery_period"
        },

        {
            "question":
                "I need to speak with "
                "a real person.",

            "expected_intent":
                "contact_human_agent"
        }
    ]

    correct = 0

    for number, test in enumerate(
        test_cases,
        start=1
    ):

        question = (
            test["question"]
        )

        expected = (
            test["expected_intent"]
        )

        results = (
            vector_store
            .similarity_search(
                question,
                k=3
            )
        )

        top_result = (
            results[0]
        )

        predicted = (
            top_result.metadata.get(
                "intent"
            )
        )

        is_correct = (
            predicted == expected
        )

        if is_correct:
            correct += 1

        print("\n" + "=" * 70)

        print(
            f"TEST {number}"
        )

        print("=" * 70)

        print(
            f"Question: {question}"
        )

        print(
            f"Expected intent: "
            f"{expected}"
        )

        print(
            f"Top retrieved intent: "
            f"{predicted}"
        )

        print(
            "Result:",
            "CORRECT"
            if is_correct
            else "INCORRECT"
        )

        print(
            "\nTop 3 retrieved examples:"
        )

        for index, document in enumerate(
            results,
            start=1
        ):

            print(
                f"{index}. "
                f"[{document.metadata.get('intent')}] "
                f"{document.metadata.get('question')}"
            )

    print("\n" + "=" * 70)

    print(
        "TOP-1 RETRIEVAL RESULT"
    )

    print("=" * 70)

    print(
        f"{correct}/{len(test_cases)} "
        f"test queries retrieved the "
        f"expected intent at rank 1."
    )


if __name__ == "__main__":
    main()