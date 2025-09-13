# TODO: Add Logger
# TODO: Wrap it up with Flask so that it can used via Web interface
import os
import pandas as pd
import yaml
from rapidfuzz import fuzz
from sf_getData import getOrders, getOSANs
from multisite_agent import build_multisite_agent, classify_dataframe
import logging, sys

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "duplicates.csv")

LOGS_DIR = os.path.join(BASE_DIR, "data", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "duplicate_detection.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")]
)
logger = logging.getLogger(__name__)

# --- Load config ---
with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

csv_file = os.path.join(BASE_DIR, config["orders_csv"])
address_threshold = config["address_similarity_threshold"]
matching_fields = config["matching_fields"]
logger.info("Matching fields configured: %s", matching_fields)

# --- Load Orders ---
excluded_statuses = ["X"]
excluded_types = ["MACD", "Account Management"]
logger.info("Exclusions: statuses=%s types=%s", excluded_statuses, excluded_types)

# Check if orders CSV exists, if not fetch from Salesforce
if not os.path.exists(csv_file):
    print(f"Orders CSV not found at {csv_file}. Fetching from Salesforce...")
    df_orders = getOrders()
else:
    logger.info("Loading Orders from CSV: %s", csv_file)
    df_orders = pd.read_csv(csv_file, parse_dates=["CreatedDate", "Contract_Signed_Date__c"])
logger.info("Loaded %d orders pre-filter", len(df_orders))

df_orders = df_orders[~df_orders["Status"].isin(excluded_statuses)].copy()
df_orders = df_orders[~df_orders["Type"].isin(excluded_types)].copy()
logger.info("After exclusions: %d orders remain", len(df_orders))
print(f"Excluded orders with Status in {excluded_statuses} and Type in {excluded_types}")
print(f"Remaining orders for duplicate comparison: {len(df_orders)}")

# --- Load OSANs ---
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
osans_csv = os.path.join(RAW_DIR, "OrderSupplierAccountNumbers.csv")
df_osans = None

# Check if OSANs CSV exists, if not fetch from Salesforce
if os.path.exists(osans_csv):
    df_osans = pd.read_csv(osans_csv)
    print(f"Loaded {len(df_osans)} Active OSANs")
else:
    print("OSANs CSV not found. Fetching from Salesforce...")
    df_osans = getOSANs()
logger.info(f"Number of OSANS for the Orders {len(df_osans)}")

# --- Separate new Orders (to check) ---
start_date = pd.to_datetime(config["start_date"])
end_date = pd.to_datetime(config["end_date"])
df_orders["CreatedDate"] = pd.to_datetime(df_orders["CreatedDate"], errors="coerce")
df_orders["Contract_Signed_Date__c"] = pd.to_datetime(df_orders["Contract_Signed_Date__c"], errors="coerce")
logger.info("Duplicate check will be performed for the Orders with these dates: start=%s end=%s", start_date.date(), end_date.date())

df_new_orders = df_orders[
    (df_orders["CreatedDate"] >= start_date) &
    (df_orders["CreatedDate"] <= end_date)
].copy()
logger.info("Number of Orders in the date range that will be compared against past Orders: %d", len(df_new_orders))
# --- Exact-match fields ---
exact_fields = [f for f, t in matching_fields.items() if t == "exact"]

# Merge on exact match key
df_all_matches = pd.merge(df_new_orders, df_orders, on=exact_fields, suffixes=('_new', '_existing'))

# Filter out self-matches
df_all_matches = df_all_matches[df_all_matches['Id_new'] != df_all_matches['Id_existing']].copy()
logger.info("Number of potential duplicate pairs based on exact field matches: %d", len(df_all_matches))
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
        logger.info("Number of Duplicates after fuzzy match on locations: %d", len(df_all_matches))

# Vectorized date range matching
if date_fields:
    for field, rule in date_fields.items():
        max_days = rule.get("max_days_diff", 0)
        df_all_matches[f'date_diff_{field}'] = (df_all_matches[f'{field}_new'] - df_all_matches[f'{field}_existing']).dt.days.abs()
    # Filter by date difference
    for field, rule in date_fields.items():
        max_days = rule.get("max_days_diff", 0)
        df_all_matches = df_all_matches[df_all_matches[f'date_diff_{field}'] <= max_days]
    logger.info("Number of duplicates after applying date field logis. Eg: Contarct Signed Date: %d", len(df_all_matches))

# --- OSAN Comparison Logic ---
if df_osans is not None and len(df_osans) > 0:
    print("Applying OSAN comparison logic...")
    
    # Create OSAN lookup dictionaries for each order
    def get_active_osans_for_order(order_id):
        order_osans = df_osans[df_osans["Order__c"] == order_id]
        return set(order_osans["SupplierAccountNumber__c"].dropna().astype(str))
    
    # Get OSANs for new and existing orders
    df_all_matches["osans_new"] = df_all_matches["Id_new"].apply(get_active_osans_for_order)
    df_all_matches["osans_existing"] = df_all_matches["Id_existing"].apply(get_active_osans_for_order)
    
    # Apply OSAN comparison rules
    def osan_comparison_valid(row):
        osans_new = row["osans_new"]
        osans_existing = row["osans_existing"]
        
        # If either order has no active OSANs, skip OSAN comparison
        if len(osans_new) == 0 or len(osans_existing) == 0:
            return True
        
        # If both orders have active OSANs, they must have at least one matching OSAN
        return len(osans_new.intersection(osans_existing)) > 0
    
    # Filter matches based on OSAN comparison
    df_all_matches = df_all_matches[df_all_matches.apply(osan_comparison_valid, axis=1)]
    
    # Clean up temporary columns
    df_all_matches = df_all_matches.drop(columns=["osans_new", "osans_existing"])
    
    print(f"After OSAN filtering: {len(df_all_matches)} potential duplicates remain")
    logger.info("Number of duplicates left after applying SAN logic: %d", len(df_all_matches))
else:
    print("Skipping OSAN comparison - no OSAN data available")

# --- Multi-site filtering (after OSAN comparison) ---
if not df_all_matches.empty:
    print(df_all_matches.head())
    print("Classifying multi-site orders to exclude them...")
    logger.info("Applying multi-site and multiple circuits logic using LLM and/or heuristics...... ")

    # Combine Order_Notes__c + Notes__c + Service_Notes__c for classification
    df_all_matches["combined_notes"] = (
            df_all_matches["Order_Notes__c_new"].fillna("") + " " +
            df_all_matches["Notes__c_new"].fillna("") + " " +
            df_all_matches["Service_Notes__c_new"].fillna("")
    )

    agent = build_multisite_agent()
    # Classify multi-site
    df_all_matches = classify_dataframe(
        agent, df_all_matches, text_col="combined_notes", batch_size=20)

    # Exclude multi-site orders
    df_all_matches = df_all_matches[df_all_matches["is_multisite"] != "yes"].copy()
    print(f"After multi-site exclusion: {len(df_all_matches)} duplicates remain")
else:
    print("No matches to classify for multi-site")
logger.info("Number of duplicates after applying mult-site and circuits logic: ",len(df_all_matches))


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
logger.info("Duplicate pairs are written to a CSV file")