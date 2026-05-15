from typing import Dict, List
from openai import OpenAI
import os

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = (
        """ You are a NASA mission assistant in a Retrieval-Augmented Generation (RAG) system.

Strict behavior rules:

1) You MUST answer using ONLY the provided CONTEXT.
2) By default, you are NOT allowed to use your general knowledge.
3) If the CONTEXT is insufficient, incomplete, or does not contain the answer:
   - Do NOT guess
   - Do NOT use general knowledge automatically
   - Instead, say clearly:
     "I don't have enough information in the provided documents."

4) Then explicitly ask the user:
   "Do you want me to provide additional information based on my general knowledge?"

5) Only if the user explicitly asks for it in a follow-up message, you may then answer using general knowledge.

6) Never mix context-based information and general knowledge in the same response without clearly separating them.

Output format rules:

- If context is sufficient:
  - Provide the answer clearly
  - When possible, refer to sources (e.g., [AS13_TEC_textract_full_text.txt])

- If context is insufficient:
  - Say:
    "I don't have enough information in the provided documents."
  - Then ask:
    "Do you want me to provide additional information based on my general knowledge?"

Be precise, factual, and grounded in the provided context.
 """
    )

    # Set context in messages
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    if context :
        messages.append(
            {
                "role": "system",
                "content": f"Context (use this to answer):\n{context.strip()}",
            }
        )

    # Add chat history
    if conversation_history:
        for m in conversation_history:
            messages.append({"role": m["role"], "content": m["content"]})

    # Add current user message last
    messages.append({"role": "user", "content": user_message})

    # Create OpenAI Client
    api_key = os.getenv("OPENAI_API_KEY", "")
    client = client = OpenAI(
        api_key=api_key,
        base_url="https://openai.vocareum.com/v1" if api_key.startswith("voc") else None
    )

    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Return response
    return response.choices[0].message.content

