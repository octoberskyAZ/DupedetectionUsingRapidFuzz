# --- Helper function to split house number ---
import re
def house_number_splitter(address):
    match = re.match(r'^(\d+)\s+(.*)', str(address))
    if match:
        return match.group(1), match.group(2)  # house_number, rest
    return None, address