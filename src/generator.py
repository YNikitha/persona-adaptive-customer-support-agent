import os

from google import genai
from dotenv import load_dotenv


load_dotenv()


class ResponseGenerator:
    """
    Generates persona-aware responses using
    Gemini and retrieved RAG context.
    """

    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )


    def get_persona_prompt(self, persona):
        """
        Return instructions based on customer persona.
        """

        prompts = {

            "Technical Expert": """
            You are a senior technical support engineer.

            Provide:
            - Detailed explanations
            - Root cause analysis
            - Technical troubleshooting steps
            - Configuration details when available
            """,

            "Frustrated User": """
            You are a calm and empathetic support agent.

            Provide:
            - Apologize for inconvenience
            - Use simple language
            - Give clear action steps
            - Maintain a reassuring tone
            """,

            "Business Executive": """
            You are a professional customer success manager.

            Provide:
            - Brief and direct responses
            - Explain business impact
            - Give resolution expectations
            - Avoid unnecessary technical details
            """
        }

        return prompts.get(
            persona,
            "You are a helpful customer support assistant."
        )


    def generate_response(
        self,
        user_query,
        persona,
        retrieved_context
    ):
        """
        Generate grounded response using RAG context.
        """

        context_text = ""

        for item in retrieved_context:

            context_text += (
                f"Source: {item['source']}\n"
                f"Information: {item['text']}\n\n"
            )


        system_prompt = f"""
        {self.get_persona_prompt(persona)}

        IMPORTANT RULES:
        1. Answer ONLY using the provided support documents.
        2. Do NOT make up information.
        3. If information is missing, say that you cannot find
           the required details.
        4. Keep the response suitable for the customer's persona.

        SUPPORT KNOWLEDGE:
        {context_text}
        """


        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_query,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.2
            }
        )

        return response.text