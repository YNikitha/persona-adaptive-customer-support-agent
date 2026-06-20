import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def classify_customer_persona(user_message: str) -> dict:
    """
    Classifies a customer message into one of three personas:
    Technical Expert, Frustrated User, or Business Executive.
    """

    # Initialize Gemini Client
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # Instructions for Gemini
    system_instruction = """
    You are a customer persona classification engine.

    Analyze the customer's message and classify it into exactly one category:

    1. Technical Expert
    - Uses technical terms
    - Asks about APIs, logs, configurations
    - Wants detailed explanations

    2. Frustrated User
    - Shows frustration or anger
    - Uses emotional or urgent language
    - Mentions repeated failures

    3. Business Executive
    - Focuses on business impact
    - Wants concise information
    - Asks about timelines or operations

    Return your response only as valid JSON.
    """

    # Define JSON response format
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "persona": {
                "type": "STRING",
                "enum": [
                    "Technical Expert",
                    "Frustrated User",
                    "Business Executive"
                ]
            },
            "confidence": {
                "type": "NUMBER"
            },
            "reason": {
                "type": "STRING"
            }
        },
        "required": [
            "persona",
            "confidence",
            "reason"
        ]
    }

    # Generate classification
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1
        )
    )

    # Convert JSON text into Python dictionary
    return json.loads(response.text)