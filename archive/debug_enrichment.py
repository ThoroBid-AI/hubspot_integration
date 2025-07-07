import google.generativeai as genai
import json
import re

# Configure Gemini
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

def test_single_query():
    """Test a single query to see what Gemini returns"""
    entity_name = "4M RANCH"
    
    prompt = f"""
Find the official contact information for this horse industry consignor: "{entity_name}"

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
    
    print(f"Testing query for: {entity_name}")
    print("Prompt:", prompt)
    print("\n" + "="*50 + "\n")
    
    try:
        response = model.generate_content(prompt)
        
        if response and response.text:
            print("Raw response:")
            print(response.text)
            print("\n" + "="*50 + "\n")
            
            # Try to parse JSON
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                print("Extracted JSON:")
                print(json_str)
                
                try:
                    parsed = json.loads(json_str)
                    print("\nParsed data:")
                    for key, value in parsed.items():
                        print(f"  {key}: {value}")
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
            else:
                print("No JSON found in response")
        else:
            print("No response received")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_single_query()