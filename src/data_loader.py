import pandas as pd
from langchain_core.documents import Document


# ============================================================
# GROCERYGO-RELEVANT INTENTS
# ============================================================

RELEVANT_INTENTS = {
    # Account
    "create_account",
    "recover_password",
    "registration_problems",
    "edit_account",
    "delete_account",
    "switch_account",

    # Orders
    "place_order",
    "change_order",
    "cancel_order",
    "track_order",

    # Delivery / Shipping
    "delivery_options",
    "delivery_period",
    "change_shipping_address",
    "set_up_shipping_address",

    # Payment
    "check_payment_methods",
    "payment_issue",

    # Refunds
    "get_refund",
    "check_refund_policy",
    "track_refund",

    # Support
    "contact_customer_service",
    "contact_human_agent",
    "complaint",
}


# ============================================================
# CONTROLLED GROCERYGO RESPONSES
# ============================================================

INTENT_RESPONSES = {
    "create_account":
        "To create a GroceryGo account, use the registration "
        "option and provide the requested account information.",

    "recover_password":
        "If you forgot your password, use the password recovery "
        "or forgot-password option on the sign-in page and follow "
        "the instructions to reset it.",

    "registration_problems":
        "If you are having trouble registering for an account, "
        "check the information you entered and try again. If the "
        "problem continues, contact customer support.",

    "edit_account":
        "You can update your account information from the account "
        "or profile settings section.",

    "delete_account":
        "If you want to delete your GroceryGo account, use the "
        "available account-management options or contact customer "
        "support for assistance.",

    "switch_account":
        "To use a different account, sign out of the current "
        "account and sign in with the account you want to use.",

    "place_order":
        "To place an order, select the items you want, add them "
        "to your cart, provide the required delivery information, "
        "choose an available payment method, and complete checkout.",

    "change_order":
        "If order changes are available, use the order-management "
        "options to modify the order. Availability may depend on "
        "the current order status.",

    "cancel_order":
        "To request an order cancellation, open your order details "
        "and use the available cancellation option. Whether an order "
        "can still be cancelled may depend on its current status.",

    "track_order":
        "You can check the current status of your order from the "
        "order tracking or order details section.",

    "delivery_options":
        "Available delivery options are shown during the ordering "
        "or checkout process.",

    "delivery_period":
        "The available delivery period or estimated delivery time "
        "is shown when arranging delivery for your order.",

    "change_shipping_address":
        "If the order status allows address changes, update the "
        "delivery address through the order or delivery settings. "
        "If you cannot change it, contact customer support.",

    "set_up_shipping_address":
        "You can provide or configure your delivery address through "
        "your account or during the checkout process.",

    "check_payment_methods":
        "The payment methods currently available to you are shown "
        "during checkout.",

    "payment_issue":
        "If you experience a payment problem, verify your payment "
        "details and try again. If the issue continues, contact "
        "customer support.",

    "get_refund":
        "You can request a refund through the appropriate order-help "
        "or customer-support option. Refund approval depends on the "
        "applicable refund policy and the details of the request.",

    "check_refund_policy":
        "Refund eligibility depends on the applicable refund policy "
        "and the circumstances of the order. Review the refund "
        "information provided by GroceryGo or contact customer support.",

    "track_refund":
        "You can check the status of a refund through the relevant "
        "order or refund information. Contact customer support if "
        "you need additional assistance.",

    "contact_customer_service":
        "You can contact GroceryGo customer support through the "
        "customer-support or help section.",

    "contact_human_agent":
        "If you need assistance from a human representative, use "
        "the customer-support or help section to request assistance.",

    "complaint":
        "If you would like to make a complaint, contact customer "
        "support and provide the relevant details so the issue can "
        "be reviewed.",
}


# ============================================================
# FRIENDLY CATEGORY NAMES
# ============================================================

CATEGORY_NAMES = {
    "ACCOUNT": "Account",
    "ORDER": "Orders",
    "DELIVERY": "Delivery",
    "SHIPPING": "Delivery",
    "PAYMENT": "Payment",
    "REFUNDS": "Refunds",
    "CONTACT": "Support",
    "FEEDBACK": "Support",
}


def load_faq_data(file_path: str):
    """
    Load and preprocess the Kaggle Bitext customer-support dataset.

    The original dataset contains customer utterances and intent
    labels rather than FAQ answers. Relevant customer-service
    intents are selected and mapped to controlled GroceryGo
    responses.

    Each customer utterance becomes a LangChain Document for
    semantic retrieval.
    """

    df = pd.read_csv(file_path)

    required_columns = {
        "flags",
        "utterance",
        "category",
        "intent"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The Bitext dataset must contain flags, utterance, "
            "category, and intent columns."
        )

    # Keep only fields needed by the chatbot.
    processed_df = df[
        ["utterance", "category", "intent"]
    ].copy()

    # Remove incomplete records.
    processed_df = processed_df.dropna(
        subset=["utterance", "category", "intent"]
    )

    # Clean text fields.
    processed_df["utterance"] = (
        processed_df["utterance"]
        .astype(str)
        .str.strip()
    )

    processed_df["category"] = (
        processed_df["category"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    processed_df["intent"] = (
        processed_df["intent"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Remove empty utterances.
    processed_df = processed_df[
        processed_df["utterance"] != ""
    ]

    # Keep only intents used by GroceryGo.
    processed_df = processed_df[
        processed_df["intent"].isin(
            RELEVANT_INTENTS
        )
    ].copy()

    # Remove duplicate customer utterances.
    processed_df = processed_df.drop_duplicates(
        subset=["utterance"]
    )

    # Add controlled response for every intent.
    processed_df["answer"] = (
        processed_df["intent"].map(
            INTENT_RESPONSES
        )
    )

    # Convert original dataset categories into
    # user-friendly GroceryGo categories.
    processed_df["display_category"] = (
        processed_df["category"].map(
            CATEGORY_NAMES
        )
    )

    processed_df["display_category"] = (
        processed_df["display_category"]
        .fillna(
            processed_df["category"]
            .str.title()
        )
    )

    processed_df = processed_df.reset_index(
        drop=True
    )

    documents = []

    for _, row in processed_df.iterrows():

        content = (
            f"Customer question: {row['utterance']}\n"
            f"Intent: {row['intent']}\n"
            f"Approved answer: {row['answer']}"
        )

        document = Document(
            page_content=content,
            metadata={
                "question": row["utterance"],
                "category": row["display_category"],
                "original_category": row["category"],
                "intent": row["intent"],
                "answer": row["answer"],
            }
        )

        documents.append(document)

    return documents, processed_df