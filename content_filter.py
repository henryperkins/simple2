import asyncio
import random
from logger import log_error, log_info

class ContentFilter:
    def __init__(self, client):
        self.client = client
        self.max_retries = 3

    async def check_content(self, text: str) -> dict:
        """
        Perform content safety check with retry logic.

        Args:
            text (str): Content to check

        Returns:
            dict: Safety check results
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.moderations.create(input=text)
                results = response.results[0]
                flagged = any(results.flagged)
                annotations = results.categories._asdict()

                if flagged:
                    log_info(f"Content flagged: {annotations}")
                else:
                    log_info("Content is safe.")

                return {'safe': not flagged, 'annotations': annotations}

            except Exception as e:
                log_error(f"Error during content safety check attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return {'safe': True, 'error': str(e)}
                await asyncio.sleep(2 ** attempt + random.uniform(0, 1))  # Exponential backoff with jitter