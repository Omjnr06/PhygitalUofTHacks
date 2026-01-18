import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = "http://localhost:3000"
APP_NAME = "Phygital Store Scanner"

def get_store_insights():
    if not OPENROUTER_API_KEY:
        print("❌ Error: API Key not found. Check your .env file.")
        return "Error: Missing API Key"

    # --- PATHS ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zones_path = os.path.join(base_dir, "assets", "zones.json")
    analytics_path = os.path.join(base_dir, "assets", "store_analytics.json")
    
    # New Output Folder
    output_folder = os.path.join(base_dir, "assets", "ai_agent_returns")
    os.makedirs(output_folder, exist_ok=True)

    insights_output_path = os.path.join(output_folder, "ai_insights.txt")
    optimized_zones_path = os.path.join(output_folder, "optimized_zones.json")

    # --- VALIDATION ---
    if not os.path.exists(analytics_path) or not os.path.exists(zones_path):
        return "Error: JSON data files not found. Run the scanner first."

    with open(zones_path, 'r') as f:
        zones_data = json.load(f)
    
    with open(analytics_path, 'r') as f:
        analytics_data = json.load(f)

    # --- AI REQUEST ---
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    print("🧠 Sending data to OpenRouter (GPT-4o)...")

    prompt = f"""
    You are an expert Retail Analyst AI. I am sending you data in the form of 2 JSONS from a physical store tracking system.
    
    FILE 1: LAYOUT (zones.json)
    {json.dumps(zones_data, indent=2)}

    FILE 2: TRAFFIC REPORT (store_analytics.json)
    {json.dumps(analytics_data, indent=2)}

    TASK:
    Analyze this data and provide a concise, actionable report AND a JSON recommendation for layout changes.
    
    PART 1: REPORT (Plain Text)
    1. Identify the "Hot Zones" vs "Cold Zones".
    2. Compare the product categories: Why might one be performing better than the other?
    3. Give 3 specific recommendations to improve store layout or sales based ONLY on this data.
    4. Give ideas on rearranging different products to possibly boost sales.
    
    PART 2: JSON OUTPUT (Strict Format)
    Based on your insights, I want you to "rearrange" the store. 
    Return a valid JSON structure under a key called "optimized_layout".
    It should look exactly like the input zones.json, but with the 'product' names swapped to their new, optimal locations.
    Keep the 'id' and 'coordinates' exactly the same, only change the 'product' string if you think it should move.
    
    Format your response exactly like this, with a separator:
    [REPORT]
    (Your text report here)
    [JSON]
    (Your JSON here)
    """

    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": YOUR_SITE_URL,
            "X-Title": APP_NAME,
        },
        model="openai/gpt-4o",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    full_response = completion.choices[0].message.content
    
    # --- PARSING ---
    if "[JSON]" in full_response:
        parts = full_response.split("[JSON]")
        text_report = parts[0].replace("[REPORT]", "").strip()
        json_part = parts[1].strip()
        
        # Clean markdown wrappers
        if json_part.startswith("```"):
            json_part = json_part.split("\n", 1)[1]
            json_part = json_part.rsplit("\n", 1)[0]
        
        # Save Text
        with open(insights_output_path, "w") as f:
            f.write(text_report)
        print(f"✅ AI Insights saved to: {insights_output_path}")

        # Save JSON
        try:
            optimized_data = json.loads(json_part)
            if "optimized_layout" in optimized_data:
                final_json = optimized_data["optimized_layout"]
            else:
                final_json = optimized_data
                
            with open(optimized_zones_path, "w") as f:
                json.dump(final_json, f, indent=4)
            print(f"✅ Optimized Layout saved to: {optimized_zones_path}")
            
        except json.JSONDecodeError:
            print("❌ Error: AI produced invalid JSON.")
    else:
        print("⚠️ Warning: AI did not use the [JSON] separator. Saving full response as text.")
        with open(insights_output_path, "w") as f:
            f.write(full_response)

    return "AI Analysis Complete"

if __name__ == "__main__":
    get_store_insights()