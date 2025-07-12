import ollama
from config import SECTOR_LIST, PRODUCT_TYPES, PRODUCT_CATEGORIES, PROMOTIONS

def parse_with_ollama(user_input):
    prompt = f"""
You are a financial product assistant. Analyze the following product/campaign description and extract these fields as a valid Python dictionary with only field names as keys:
- segment (choose from: Individual, SME, Corporate)
- sector (choose from: {SECTOR_LIST})
- product_type (choose from: {PRODUCT_TYPES})
- product_category (choose from: {PRODUCT_CATEGORIES})
- promotion (choose from: {PROMOTIONS})
- channel (Digital, Branch, Both)
- term (integer, in months)
- interest_type (Fixed, Variable, None)
- risk_level (High, Medium, Low)
- innovation_level (High, Medium, Low)
- launch_year (integer)
If a field is not mentioned, use None or an empty list.

Description: \"\"\"{user_input}\"\"\"
Return only the valid Python dictionary, no explanation.
"""
    response = ollama.generate(
        model="llama3",  # or your preferred local model
        prompt=prompt,
        stream=False
    )
    import ast
    try:
        parsed_fields = ast.literal_eval(response['response'].strip())
    except Exception:
        parsed_fields = {}
    return parsed_fields
