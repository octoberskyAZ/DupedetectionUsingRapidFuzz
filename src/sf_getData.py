import pandas as pd
from sf_connection import getSFConnection
from datetime import datetime, timedelta
import os

RAW_ORDERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

def getOrders():
    sf = getSFConnection()
    six_months_ago = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')

    # SOQL query
    query = f"""
        SELECT Id, OrderNumber, Telarus_Order_ID__c, Partner_Name__c, AccountId, Supplier__c, Address__c, 
        Status, Type, Is_Base_Transfer__c, Order_Notes__c, Notes__c, Service_Notes__c,Contract_Signed_Date__c, MRC__c, ScoreboardMRC__c, ScoreboardDate__c, CreatedDate
        FROM Order
        WHERE CreatedDate >= {six_months_ago}T00:00:00Z Order By Partner_Name__c, AccountId, CreatedDate
    """

    print("Querying Salesforce...")
    result = sf.query_all(query)
    records = result['records']

    df = pd.DataFrame(records).drop(columns='attributes')
    csv_file = os.path.join(RAW_ORDERS_DIR, 'Orders_last_6_months.csv')
    df.to_csv(csv_file, index=False)
    print(f"Saved {len(df)} orders to {csv_file}")

    return df

def getOrdersdf():
    orders_csv_path = os.path.join(RAW_ORDERS_DIR, 'Orders_last_6_months.csv')
    if not os.path.exists(orders_csv_path):
        raise FileNotFoundError(f"Orders CSV not found at {orders_csv_path}. Run getOrders() first or place the file there.")
    
    orders_df = pd.read_csv(orders_csv_path, dtype=str)
    if 'Id' not in orders_df.columns:
        raise ValueError("'Id' column not found in Orders CSV. Ensure the CSV has an 'Id' column.")

    order_ids = orders_df['Id'].dropna().unique().tolist()
    if not order_ids:
        print("No Order IDs found in CSV. Nothing to query.")
    return order_ids

def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def getOSANs():
    sf = getSFConnection()
    
    order_ids = getOrdersdf()
    print(f"Querying OrderSupplierAccountNumber for {len(order_ids)} Order IDs in batches...")

    all_records = []
    for batch_num, batch_ids in enumerate(chunk_list(order_ids, 500), start=1):
        in_clause = ", ".join([f"'{oid}'" for oid in batch_ids])
        query = f"""
            SELECT Id, Order__c, Supplier__c, SupplierAccountNumber__c, Status__c
            FROM OrderSupplierAccountNumber__c
            WHERE Order__c IN ({in_clause}) AND Status__c = 'Active'
        """
        result = sf.query_all(query)
        records = result.get('records', [])
        all_records.extend(records)
        print(f"Batch {batch_num}: fetched {len(records)} records")

    if not all_records:
        print("No OrderSupplierAccountNumber records found for provided Order IDs.")
        return pd.DataFrame()

    osans_df = pd.DataFrame(all_records)
    if 'attributes' in osans_df.columns:
        osans_df = osans_df.drop(columns=['attributes'])

    output_csv = os.path.join(RAW_ORDERS_DIR, 'OrderSupplierAccountNumbers.csv')
    osans_df.to_csv(output_csv, index=False)
    print(f"Saved {len(osans_df)} records to {output_csv}")

    return osans_df

if __name__ == "__main__":
    # Reads existing Orders CSV and fetches related OrderSupplierAccountNumber rows
    getOSANs()


