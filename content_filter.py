import asyncio
from logger import log_info, log_error

class ContentFilter:
    """
    Implements content filtering to ensure safe and appropriate AI-generated content.
    """
    def __init__(self, client):
        self.client = client
        self.blocked_terms = set()
        self.content_categories = {
            "hate": 0.7,
            "sexual": 0.8,
            "violence": 0.8,
            "self-harm": 0.9
        }

    def add_blocked_terms(self, terms):
        self.blocked_terms.update(terms)

    async def check_content(self, text):
        # Check against blocked terms
        for term in self.blocked_terms:
            if term.lower() in text.lower():
                return {"safe": False, "reason": f"Blocked term: {term}"}

        # Use Azure OpenAI content filtering
        try:
            response = await self.client.moderations.create(input=text)
            results = response.results[0]

            for category, threshold in self.content_categories.items():
                if getattr(results.categories, category) > threshold:
                    return {
                        "safe": False,
                        "reason": f"Content filtered: {category}"
                    }

            return {"safe": True, "reason": None}
        except Exception as e:
            log_error(f"Content filtering error: {e}")
            return {"safe": False, "reason": "Error in content check"}