import logging
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    logger.debug("Creating Azure OpenAI client...")
    client = AzureOpenAI(
        api_key="40c4befe4179411999c14239f386e24d",
        api_version="2024-08-01-preview",
        azure_endpoint="https://openai-hp.openai.azure.com"
    )
    logger.debug("Client created successfully")

    logger.debug("Preparing chat completion request...")
    response = client.chat.completions.create(
        model="gpt-4o",  # Correctly specify the model or deployment name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ],
        temperature=0.7,
        max_tokens=150
    )
    logger.debug("Response received successfully")
    
    print("\nResponse content:")
    print(response.choices[0].message.content)

except Exception as e:
    logger.error(f"An error occurred: {type(e).__name__}")
    logger.error(f"Error message: {str(e)}")
