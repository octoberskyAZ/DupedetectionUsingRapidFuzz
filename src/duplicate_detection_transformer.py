import os
import pandas as pd
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils import house_number_splitter

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "duplicates_transformer.csv")

# --- Load config ---
with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

csv_file = os.path.join(BASE_DIR, config["orders_csv"])
matching_fields = config["matching_fields"]
excluded_statuses = config.get("excluded_statuses", ["X"])
fuzzy_threshold = config.get("default_fuzzy_threshold", 90)

# --- Load Orders ---
df_orders = pd.read_csv(csv_file, parse_dates=["CreatedDate", "Contract_Signed_Date__c"])
df_orders = df_orders[~df_orders["Status"].isin(excluded_statuses)].copy()

# --- Split new Orders ---
start_date = pd.to_datetime(config["start_date"])
end_date = pd.to_datetime(config["end_date"])
df_new_orders = df_orders[(df_orders["CreatedDate"] >= start_date) & (df_orders["CreatedDate"] <= end_date)].copy()

# --- Exact-match fields ---
exact_fields = [f for f, t in matching_fields.items() if t == "exact"]

# Filter df_orders to only those matching exact fields with new Orders
if exact_fields:
    merge_cols = exact_fields
    df_all_matches = pd.merge(df_new_orders, df_orders, on=merge_cols, suffixes=('_new', '_existing'))
else:
    # If no exact-match fields, compare all new orders to all orders
    df_all_matches = df_new_orders.assign(key=1).merge(df_orders.assign(key=1), on='key', suffixes=('_new', '_existing')).drop('key', axis=1)

# Filter out self-matches
df_all_matches = df_all_matches[df_all_matches['Id_new'] != df_all_matches['Id_existing']].copy()

# --- Fuzzy / text similarity fields ---
fuzzy_fields = {f: rule for f, rule in matching_fields.items() if rule == "fuzzy"}
if fuzzy_fields:
    model = SentenceTransformer('all-MiniLM-L6-v2')

    for field in fuzzy_fields:
        print(f"Encoding Orders for field: {field}")

        # Preprocess with house_number_splitter
        df_all_matches[f'{field}_new_split'] = df_all_matches[f'{field}_new'].astype(str).apply(house_number_splitter)
        df_all_matches[f'{field}_existing_split'] = df_all_matches[f'{field}_existing'].astype(str).apply(house_number_splitter)

        # Fill None values with empty string to avoid encoder errors
        df_all_matches[f'{field}_new_split'] = df_all_matches[f'{field}_new_split'].fillna('')
        df_all_matches[f'{field}_existing_split'] = df_all_matches[f'{field}_existing_split'].fillna('')

        # Get unique split texts
        new_texts = df_all_matches[f'{field}_new_split'].astype(str).drop_duplicates().tolist()
        existing_texts = df_all_matches[f'{field}_existing_split'].astype(str).drop_duplicates().tolist()

        # Encode embeddings
        new_embeddings = model.encode(new_texts, show_progress_bar=True, convert_to_numpy=True)
        existing_embeddings = model.encode(existing_texts, show_progress_bar=True, convert_to_numpy=True)

        # Create lookup dicts
        new_map = dict(zip(new_texts, new_embeddings))
        existing_map = dict(zip(existing_texts, existing_embeddings))

        # Map embeddings back to dataframe
        df_all_matches['new_emb'] = df_all_matches[f'{field}_new_split'].map(new_map)
        df_all_matches['existing_emb'] = df_all_matches[f'{field}_existing_split'].map(existing_map)

        # Drop rows where either embedding is missing
        df_all_matches = df_all_matches[df_all_matches['new_emb'].notna() & df_all_matches['existing_emb'].notna()]

        # Compute cosine similarity in batch
        sim_scores = np.array([
            cosine_similarity(np.array([n]), np.array([e]))[0][0]
            for n, e in zip(df_all_matches['new_emb'], df_all_matches['existing_emb'])
        ])

        df_all_matches[f'sim_{field}'] = sim_scores * 100  # 0-100 scale
        df_all_matches = df_all_matches[df_all_matches[f'sim_{field}'] >= fuzzy_threshold]

        # Drop temporary embedding columns
        df_all_matches = df_all_matches.drop(columns=['new_emb', 'existing_emb'])

# --- Date range fields ---
date_fields = {f: rule for f, rule in matching_fields.items() if isinstance(rule, dict) and rule.get("type") == "date_range"}
for field, rule in date_fields.items():
    max_days = rule.get("max_days_diff", 0)
    df_all_matches[f'date_diff_{field}'] = (df_all_matches[f'{field}_new'] - df_all_matches[f'{field}_existing']).dt.days.abs()
    df_all_matches = df_all_matches[df_all_matches[f'date_diff_{field}'] <= max_days]

# --- Extract duplicates ---
df_duplicates = df_all_matches[['OrderNumber_new', 'OrderNumber_existing']].copy()
df_duplicates.columns = ['New_Order_Number', 'Existing_Order_Number']

# Ensure the larger OrderNumber is always "New"
df_duplicates[['New_Order_Number', 'Existing_Order_Number']] = df_duplicates.apply(
    lambda row: [row['New_Order_Number'], row['Existing_Order_Number']]
    if row['New_Order_Number'] >= row['Existing_Order_Number']
    else [row['Existing_Order_Number'], row['New_Order_Number']],
    axis=1,
    result_type='expand'
)

df_duplicates = df_duplicates.drop_duplicates()
df_duplicates.to_csv(OUTPUT_FILE, index=False)
print(f"Found {len(df_duplicates)} potential duplicates. Saved to {OUTPUT_FILE}")
