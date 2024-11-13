import aiohttp
import asyncio
import json
import os
import sentry_sdk
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from logger import logger  # Use the global logger instance
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Core configuration
endpoint = "https://openai-hp.openai.azure.com/"
deployment = "gpt-4o"
model_name = "gpt-4o-2024-08-06"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY") or ""

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-08-01-preview"
)

async def make_openai_request(
    messages: list, service: str, model_name: Optional[str] = None
) -> Any:
    logger.info(f"Preparing to make request to {service} service")
    
    if service == "azure":
        model_name = deployment
    else:
        model_name = "text-davinci-003"

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.13,
        "max_tokens": 16384,
        "top_p": 0.95
    }

    logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

    retries = 3
    base_backoff = 2

    for attempt in tqdm(range(1, retries + 1), desc="API Request Progress"):
        logger.info(f"Attempt {attempt} of {retries}")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            logger.debug(f"Received response: {response}")
            return response
        except Exception as e:
            logger.error(f"Unexpected exception during API request: {e}")
            sentry_sdk.capture_exception(e)

        sleep_time = base_backoff ** attempt
        logger.debug(f"Retrying API request in {sleep_time} seconds (Attempt {attempt}/{retries})")
        await asyncio.sleep(sleep_time)

    logger.error("Exceeded maximum retries for API request.")
    return {"error": "Failed to get a successful response from the API."}

async def analyze_function_with_openai(
    function_details: Dict[str, Any], service: str
) -> Dict[str, Any]:
    function_name = function_details.get("name", "unknown")
    logger.info(f"Analyzing function: {function_name}")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates documentation.",
        },
        {
            "role": "user",
            "content": (
                f"Provide a detailed analysis for the following function:\n\n"
                f"{function_details.get('code', '')}"
            ),
        },
    ]

    try:
        # Include actual function details in the API request
        response = await make_openai_request(
            messages=messages,
            service=service,
        )

        if "error" in response:
            logger.error(f"API returned an error: {response['error']}")
            sentry_sdk.capture_message(f"API returned an error: {response['error']}")
            return {
                "name": function_name,
                "complexity_score": function_details.get("complexity_score", "Unknown"),
                "summary": "Error during analysis.",
                "docstring": "Error: Documentation generation failed.",
                "changelog": "Error: Changelog generation failed.",
            }

        choices = response.get("choices", [])
        if not choices:
            error_msg = "Missing 'choices' in API response."
            logger.error(error_msg)
            raise KeyError(error_msg)

        response_message = choices[0].get("message", {})
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": response_message.get("content", "No summary available."),
            "docstring": "No docstring available.",
            "changelog": "No changelog available.",
        }

    except (KeyError, TypeError, json.JSONDecodeError) as e:
        logger.error(f"Error processing API response: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": "Error during analysis.",
            "docstring": "Error: Documentation generation failed.",
            "changelog": "Error: Changelog generation failed.",
        }

    except Exception as e:
        logger.error(f"Unexpected error during function analysis: {e}")
        sentry_sdk.capture_exception(e)
        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": "Error during analysis.",
            "docstring": "Error: Documentation generation failed.",
            "changelog": "Error: Changelog generation failed.",
        }
