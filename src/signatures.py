import dspy
from .config import VALID_CATEGORIES
from typing import Literal

EmailCategory = Literal[tuple(VALID_CATEGORIES)]

class EmailClassificationSignature(dspy.Signature):
    """
    Signature for classifying emails into predefined categories.
    """
    email_content: str = dspy.InputField(desc="The entire content of the email, including subject and body.")
    category: EmailCategory = dspy.OutputField(desc=f"The classification category of the email. Must be one of: {', '.join(VALID_CATEGORIES)}")

class ResponseGenerationSignature(dspy.Signature):
    """
    Signature for generating automated responses to emails.
    """
    email_content: str = dspy.InputField(desc="The entire content of the email, including subject and body.")
    category: str = dspy.InputField(desc=f"The category assigned to the email.")
    response: str = dspy.OutputField(desc="A brief, helpful automated response to the email suitable for the given category.")
