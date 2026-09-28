from src.data_loader import (
    load_faq_data
)


DATASET_PATH = (
    "bitext_customer_support.csv"
)


documents, df = load_faq_data(
    DATASET_PATH
)


print("=" * 70)
print("KAGGLE DATASET PREPROCESSING TEST")
print("=" * 70)


print("\nProcessed dataset loaded successfully!")


print(
    "\nNumber of processed records:",
    len(df)
)


print(
    "Number of processed features:",
    len(df.columns)
)


print(
    "Processed features:",
    list(df.columns)
)


print("\nGroceryGo Categories:")

print(
    df["display_category"]
    .value_counts()
)


print("\nSelected Intents:")

print(
    df["intent"]
    .value_counts()
)


print(
    "\nNumber of LangChain documents:",
    len(documents)
)


print(
    "\nFirst LangChain document:"
)

print(
    documents[0]
)