import os
import pandas as pd
import yaml
from rapidfuzz import fuzz

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "duplicates.csv")

# --- Load config ---
with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

csv_file = os.path.join(BASE_DIR, config["orders_csv"])
address_threshold = config["address_similarity_threshold"]
matching_fields = config["matching_fields"]

# --- Load Orders ---
excluded_statuses = ["X"]
df_orders = pd.read_csv(csv_file, parse_dates=["CreatedDate", "Contract_Signed_Date__c"])
df_orders = df_orders[~df_orders["Status"].isin(excluded_statuses)].copy()

# --- Separate new Orders (to check) ---
start_date = pd.to_datetime(config["start_date"])
end_date = pd.to_datetime(config["end_date"])
df_new_orders = df_orders[
    (df_orders["CreatedDate"] >= start_date) &
    (df_orders["CreatedDate"] <= end_date)
].copy()

# --- Exact-match fields ---
exact_fields = [f for f, t in matching_fields.items() if t == "exact"]

# Merge on exact match key
df_all_matches = pd.merge(df_new_orders, df_orders, on=exact_fields, suffixes=('_new', '_existing'))

# Filter out self-matches
df_all_matches = df_all_matches[df_all_matches['Id_new'] != df_all_matches['Id_existing']].copy()

# --- Apply fuzzy and date range matching ---
fuzzy_fields = {f: rule for f, rule in matching_fields.items() if rule == "fuzzy"}
date_fields = {f: rule for f, rule in matching_fields.items() if isinstance(rule, dict) and rule.get("type") == "date_range"}

# Vectorized fuzzy matching
if fuzzy_fields:
    for field, rule in fuzzy_fields.items():
        # If rule is a dict, use its "threshold"; otherwise fall back to global default
        threshold = rule.get("threshold", config.get("default_fuzzy_threshold", 90)) if isinstance(rule, dict) else config.get("default_fuzzy_threshold", 90)

        df_all_matches[f'fuzzy_match_{field}'] = df_all_matches.apply(
            lambda row:
            # Check for a house number mismatch first
            0 if str(row[f'{field}_new']).split()[0] != str(row[f'{field}_existing']).split()[0]
            # If house numbers match, proceed with fuzzy matching
            else fuzz.token_set_ratio(str(row[f'{field}_new']), str(row[f'{field}_existing'])),
            axis=1
        )

        # Apply threshold specific to this field
        df_all_matches = df_all_matches[df_all_matches[f'fuzzy_match_{field}'] >= threshold]

# Vectorized date range matching
if date_fields:
    for field, rule in date_fields.items():
        max_days = rule.get("max_days_diff", 0)
        df_all_matches[f'date_diff_{field}'] = (df_all_matches[f'{field}_new'] - df_all_matches[f'{field}_existing']).dt.days.abs()
    # Filter by date difference
    for field, rule in date_fields.items():
        max_days = rule.get("max_days_diff", 0)
        df_all_matches = df_all_matches[df_all_matches[f'date_diff_{field}'] <= max_days]

# --- Extract and save duplicates ---
df_duplicates = df_all_matches[['OrderNumber_new', 'OrderNumber_existing']].copy()
df_duplicates.columns = ['New_Order_Number', 'Existing_Order_Number']

# Ensure the larger OrderNumber is always treated as "New"
df_duplicates[['New_Order_Number', 'Existing_Order_Number']] = df_duplicates.apply(
    lambda row: [row['New_Order_Number'], row['Existing_Order_Number']]
    if row['New_Order_Number'] >= row['Existing_Order_Number']
    else [row['Existing_Order_Number'], row['New_Order_Number']],
    axis=1,
    result_type='expand'
)

# Drop exact duplicate rows
df_duplicates = df_duplicates.drop_duplicates()

df_duplicates.to_csv(OUTPUT_FILE, index=False)
print(f"Found {len(df_duplicates)} potential duplicates. Saved to {OUTPUT_FILE}")