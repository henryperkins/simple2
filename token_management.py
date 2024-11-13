from tiktoken import encoding_for_model
from logger import log_info

def estimate_tokens(text, model="gpt-4"):
    encoding = encoding_for_model(model)
    return len(encoding.encode(text))

def optimize_prompt(text, max_tokens=4000):
    current_tokens = estimate_tokens(text)
    if current_tokens > max_tokens:
        # Truncate the prompt
        encoding = encoding_for_model("gpt-4")
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        log_info("Prompt optimized to fit within token limits.")
        return truncated_text
    return text