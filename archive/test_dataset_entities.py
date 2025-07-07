import pandas as pd
import google.generativeai as genai
import json
import re

# Load the cleaned data to find some better entities
df = pd.read_csv('/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_cleaned.csv')

# Get some entities that might be more well-known
consignors = df['consignor_cleaned'].value_counts().head(10)
buyers = df['buyer_cleaned'].value_counts().head(10)

print("Top consignors:")
print(consignors)
print("\nTop buyers:")
print(buyers)

# Let's test with Taylor Made which appears in the data
api_key = "AIzaSyBB2-SGq0Pjbp4QDEBLVcejzz0LWg1O3_Q"
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    'gemini-2.0-flash-exp',
    generation_config=genai.GenerationConfig(
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        max_output_tokens=500
    )
)

def test_entity(entity_name, entity_type):
    prompt = f"""
Search the web and find the official contact information for this horse industry {entity_type}: "{entity_name}"

Please provide ONLY the following information in this exact JSON format:
{{
    "website": "official website URL or null",
    "email": "official email address or null", 
    "phone": "official phone number or null",
    "city": "city name or null",
    "state": "US state abbreviation (e.g., KY, FL) or null",
    "country": "country name or null",
    "is_company": true/false
}}

Rules:
- Return "null" (not empty string) if information not found
- Use web search to find current, accurate information
- Only return official/verified contact details
- For phone numbers, use US format if available
- For states, use 2-letter abbreviations (KY, FL, CA, etc.)
- is_company should be true for businesses/farms/stables, false for individuals
- Do not hallucinate or guess information
"""
    
    print(f"\n{'='*50}")
    print(f"Testing: {entity_name}")
    print('='*50)
    
    try:
        response = model.generate_content(prompt)
        
        if response and response.text:
            print("Raw response:")
            print(response.text)
            
            # Try to parse JSON
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    parsed = json.loads(json_str)
                    print("\nParsed data:")
                    for key, value in parsed.items():
                        print(f"  {key}: {value}")
                    return parsed
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
        else:
            print("No response received")
            
    except Exception as e:
        print(f"Error: {e}")
    
    return None

# Test a few entities from our dataset
test_entities = [
    "TAYLOR MADE SALES AGENCY",
    "CLAIBORNE FARM", 
    "THREE CHIMNEYS FARM"
]

for entity in test_entities:
    test_entity(entity, "consignor")