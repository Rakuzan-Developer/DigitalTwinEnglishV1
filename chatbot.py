#chatbot.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# PROMOTIONS listesini buraya da çekiyoruz (main.py'de import ettiğin için çakışmaz, burada da tanımlanabilir)
PROMOTIONS = [
    "Cashback", "Low Interest", "Free EFT", "Loyalty Points", "High Limit", "Digital Convenience",
    "Early Payment", "Extra Campaign", "Fast Approval", "Low Commission", "Extra Bonus", "Free Insurance"
]

def parse_with_mistral(user_input):
    prompt = f"""
Given the following product/campaign description, extract ALL below filters and output ONLY a valid Python dictionary, even if some fields are empty (use [] for empty lists, '' for empty strings, and 0 for integers). Use these exact keys:
- segment (list: Individual, SME, Corporate)
- sector (list)
- product_type (string)
- product_category (string)
- promotion (list, select from: {PROMOTIONS})
- channel (string)
- term (int, months)
- interest_type (string: Fixed, Variable, None)
- risk_level (string: High, Medium, Low)
- innovation_level (string: High, Medium, Low)
- launch_year (int)
Example output:
{{'segment': ['Individual'], 'sector': ['Retail'], 'product_type': 'Loan', 'product_category': 'Consumer', 'promotion': ['Cashback', 'Loyalty Points'], 'channel': 'Digital', 'term': 12, 'interest_type': 'Fixed', 'risk_level': 'Medium', 'innovation_level': 'High', 'launch_year': 2023}}
If any field is missing in the input, fill it with the appropriate empty value as shown above.

Text:
{user_input}
Output:
"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 600
    }
    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} {response.text}")
    content = response.json()['choices'][0]['message']['content']
    try:
        result = eval(content.strip())
    except Exception:
        import re
        dict_text = re.findall(r'\{.*\}', content, re.DOTALL)
        if dict_text:
            try:
                result = eval(dict_text[0])
            except Exception:
                result = {}
        else:
            result = {}

    # Fallback - tüm alanları doldur
    defaults = {
        'segment': [],
        'sector': [],
        'product_type': '',
        'product_category': '',
        'promotion': [],
        'channel': '',
        'term': 0,
        'interest_type': '',
        'risk_level': '',
        'innovation_level': '',
        'launch_year': 0,
    }
    for k, v in defaults.items():
        if k not in result:
            result[k] = v

    return result
