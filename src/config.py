import os
import logging
from dotenv import load_dotenv
import dspy

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#DSPy config
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OpenAI API key not found in environment variables.")
    raise ValueError("OpenAI API key not found.")

try:
    llm = dspy.LM(model="openai/gpt-4o", api_key=api_key, temperature=0.0, max_tokens=250)
    dspy.settings.configure(lm=llm)
    logger.info("DSPy LLM configured successfully.")

except Exception as e:
    logger.error(f"Failed to configure DSPy LLM: {e}")
    raise ValueError("Failed to configure DSPy LLM.")


DATA_FILE_PATH = "data/sample_emails.json"

VALID_CATEGORIES = {
    "complaint", "inquiry", "feedback",
    "support_request", "other"
}
