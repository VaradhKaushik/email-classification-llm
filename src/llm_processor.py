# EmailProcessor from the original given code
import dspy
from typing import Dict, Optional
from .signatures import EmailClassificationSignature, ResponseGenerationSignature   #TODO
from .config import VALID_CATEGORIES, logger


class LLMProcessor:
    def __init__(self):
        # Define valid categories
        self.valid_categories = VALID_CATEGORIES

        try:
            self.classifier = dspy.Predict(EmailClassificationSignature)
            self.responder = dspy.Predict(ResponseGenerationSignature)
            logger.info("LLMProcessor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLMProcessor: {e}")
            raise ValueError("Failed to initialize LLMProcessor.")


    def _prepare_email_content(self, email: Dict) -> str:
        """Combine subject and body into a single string."""
        subject = email.get("subject", "")
        body = email.get("body", "")
        return f"Subject: {subject}\n\nBody: {body}"
    
    
    def classify_email(self, email: Dict) -> Optional[str]:
        """
        Classify an email using LLM.
        Returns the classification category or None if classification fails.
        """
        if not self.classifier:
            logger.error("Classifier not initialized.")
            return None
        
        if not email or 'body' not in email:
            logger.warning(f"Invalid email data for classification. Email ID: {email.get('id', 'N/A')}")
            return None
        
        email_content = self._prepare_email_content(email)

        try:
            prediction = self.classifier(email_content=email_content)
            category = prediction.category

            if category in self.valid_categories:
                logger.info(f"Email {email.get('id', 'N/A')} classified as: {category}")
                return category
            else:
                logger.warning(f"Unable to classify email {email.get('id', 'N/A')}. Defaulting to 'other'.")
                return "other"

        except Exception as e:
            logger.error(f"Failed to classify email {email.get('id', 'N/A')} encountered error: {e}")
            return None


    def generate_response(self, email: Dict, classification: str) -> Optional[str]:
        """
        Generate an automated response based on email classification.
        """
        if not self.responder:
            logger.error("Responder not initialized.")
            return None
        
        if not email or not classification:
            logger.warning(f"Invalid email data or classification for response generation. Email ID: {email.get('id', 'N/A')}")
            return None
        
        email_content = self._prepare_email_content(email)

        try:
            prediction = self.responder(email_content=email_content, category=classification)
            response = prediction.response

            if response:
                logger.info(f"Generated response for email {email.get('id', 'N/A')}: {response}")
                return response
            else:
                logger.warning(f"No response generated for email {email.get('id', 'N/A')}.")
                return None
        
        except Exception as e:
            logger.error(f"Failed to generate response for email {email.get('id', 'N/A')}: {e}", exc_info=True)
            return None