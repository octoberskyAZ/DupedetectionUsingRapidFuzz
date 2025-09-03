from dotenv import load_dotenv
import os
from simple_salesforce import Salesforce

def getSFConnection():
    load_dotenv()  # Load environment variables from .env file

    username = os.getenv('SALESFORCE_USERNAME')
    password = os.getenv('SALESFORCE_PASSWORD')
    security_token = os.getenv('SALESFORCE_SECURITY_TOKEN')
    domain = os.getenv('SALESFORCE_DOMAIN')

    sf = Salesforce(username=username, password=password, security_token=security_token, domain=domain)
    return sf