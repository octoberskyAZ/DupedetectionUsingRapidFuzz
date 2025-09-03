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
        SELECT Id, OrderNumber, Telarus_Order_ID__c, Partner_Name__c, AccountId, Supplier__c, Address__c, Status, Contract_Signed_Date__c, MRC__c, ScoreboardMRC__c, ScoreboardDate__c, CreatedDate
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


if __name__ == "__main__":
    getOrders()


