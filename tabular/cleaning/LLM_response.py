import time

from openai import OpenAI


# Send a prompt to an OpenAI-compatible LLM endpoint.
def call_llm(prompt: str, base_url: str, api_key: str, model: str) -> str:
    """Call an OpenAI-compatible chat completion endpoint.

    Args:
        prompt: User prompt.
        base_url: OpenAI-compatible API base URL.
        api_key: API key.
        model: Model name.

    Returns:
        Assistant response content, or an error message after 3 failed attempts.
    """
    last_error = None

    for attempt in range(3):
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(1)

    return f"Maximum 3 retries reached for OpenAI API requests. Error: {str(last_error)}"
