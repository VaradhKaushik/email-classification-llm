from typing import Dict
from src.llm_processor import LLMProcessor
from .config import logger


class EmailAutomationSystem:
    def __init__(self, processor: LLMProcessor):
        """Initialize the automation system with an EmailProcessor."""

        if not isinstance(processor, LLMProcessor):
            raise TypeError("Processor must be an instance of LLMProcessor.")

        self.processor = processor
        self.response_handlers = {
            "complaint": self._handle_complaint,
            "inquiry": self._handle_inquiry,
            "feedback": self._handle_feedback,
            "support_request": self._handle_support_request,
            "other": self._handle_other
        }

        logger.info("EmailAutomationSystem initialized successfully.")

    def process_email(self, email: Dict) -> Dict:
        """
        Process a single email through the complete pipeline.
        Returns a dictionary with the processing results.
        
        TODO:
        1. Implement the complete processing pipeline
        2. Add appropriate error handling
        3. Return processing results
        """
        email_id = email.get("id", "N/A")
        logger.info(f"Processing email {email_id}...")

        result = {
            "email_id": email_id,
            "success": False,
            "classification": None,
            "response_sent": False,
            "action_taken": None,
            "error": None,
        }
        
        
        try:
            # Classify the email
            classification = self.processor.classify_email(email)
            
            if not classification:
                result["error"] = "Classification failed"
                flag_for_manual_review(email_id, "Classification failed")
                logger.error(f"Email {email_id} classification failed.")
                result["action_taken"] = "Flagged for manual review (Classification failed)"
                return result
            
            result["classification"] = classification
            result["success"] = True
            
            # Get response for the email
            response = self.processor.generate_response(email, classification)

            if not response:
                logger.warning(f"Response generation failed for email {email_id}.")
                response = "Automated response generation failed. Please contact support."
            else:
                result["response_sent"] = True

            # Handle the email based on its classification
            handler = self.response_handlers.get(classification, self._handle_other)

            if handler:
                action_description = handler(email, response)
                result["action_taken"] = action_description
            
            else:
                logger.error(f"No handler found for classification '{classification}' for the email {email_id}.")
                flag_for_manual_review(email_id, "No handler found")
                
                result["action_taken"] = "Flagged for manual review (No handler found)"
                result["error"] = "No handler found for classification"
                return result

           
            logger.info(f"Email {email_id} processed successfully.")


        except Exception as e:
            logger.error(f"Error processing email {email_id}: {e}", exc_info=True)
            result["error"] = str(e)
            flag_for_manual_review(email_id, str(e))
            result["action_taken"] = "Flagged for manual review (Processing error)"
            
        return result


    def _handle_complaint(self, email: Dict):
        """
        Handle complaint emails.
        """
        email_id = email.get("id", "N/A")
        context = f"Subject: {email.get('subject', 'N/A')}, Body: {email.get('body', 'N/A')}"
        send_complaint_response(email_id, "We have received your complaint and will address it shortly.")
        create_urgent_ticket(email_id, "complaint", context)
        return "Complaint handled and URGENT ticket created."

    def _handle_inquiry(self, email: Dict):
        """
        Handle inquiry emails.
        """
        email_id = email.get("id", "N/A")
        send_standard_response(email_id, "Thank you for your inquiry. We will get back to you soon.")
        return "Inquiry handled and standard response sent."

    def _handle_feedback(self, email: Dict):
        """
        Handle feedback emails.
        """
        email_id = email.get("id", "N/A")
        feedback = email.get("body", "No feedback provided.")
        log_customer_feedback(email_id, feedback)
        send_standard_response(email_id, "Thank you for your feedback!")
        return "Feedback handled and standard response sent."


    def _handle_support_request(self, email: Dict):
        """
        Handle support request emails.
        TODO: Implement support request handling logic
        """
        email_id = email.get("id", "N/A")
        context = f"Subject: {email.get('subject', 'N/A')}, Body: {email.get('body', 'N/A')}"
        create_support_ticket(email_id, context)
        send_standard_response(email_id, "Your support request has been received.")
        return "Support request handled and ticket created."

    def _handle_other(self, email: Dict):
        """
        Handle other category emails.
        """
        email_id = email.get("id", "N/A")
        flag_for_manual_review(email_id, "Other category email")
        send_standard_response(email_id, "Thank you for your email. We will review it shortly.")
        return "Other category handled and standard response sent."

# Mock service functions
def send_complaint_response(email_id: str, response: str):
    """Mock function to simulate sending a response to a complaint"""
    logger.info(f"Sending complaint response for email {email_id}, response: {response}")
    # In real implementation: integrate with email service


def send_standard_response(email_id: str, response: str):
    """Mock function to simulate sending a standard response"""
    logger.info(f"Sending standard response for email {email_id}, response: {response}")
    # In real implementation: integrate with email service


def create_urgent_ticket(email_id: str, category: str, context: str):
    """Mock function to simulate creating an urgent ticket"""
    logger.info(f"Creating urgent ticket for email {email_id} => Category: {category} => Context: {context}")
    # In real implementation: integrate with ticket system


def create_support_ticket(email_id: str, context: str):
    """Mock function to simulate creating a support ticket"""
    logger.info(f"Creating support ticket for email {email_id}, context: {context}")
    # In real implementation: integrate with ticket system


def log_customer_feedback(email_id: str, feedback: str):
    """Mock function to simulate logging customer feedback"""
    logger.info(f"Logging feedback for email {email_id}, feedback: {feedback}")
    # In real implementation: integrate with feedback system

def flag_for_manual_review(email_id: str, reason: str):
    """Mock function to simulate flagging an email for manual review"""
    logger.info(f"Flagging email {email_id} for manual review: {reason}")
    # In real implementation: integrate with review system



