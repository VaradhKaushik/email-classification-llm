import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from src.data_loader import load_email_data
from src.llm_processor import LLMProcessor
from src.automation import EmailAutomationSystem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def run_demonstration():
    """Run a demonstration of the complete system."""
    logger.info("Starting the email processing demonstration...")
    # Load data
    sample_emails = load_email_data()

    if not sample_emails:
        logger.error("No sample emails to process.")
        return None


    # Initialize the system
    try:
        processor = LLMProcessor()

        if not processor.classifier or not processor.responder:
            logger.error("Failed to initialize LLMProcessor.")
            return None
    
    except Exception as e:
        logger.error(f"Failed to initialize LLMProcessor: {e}. Exiting...")
        return None


    automation_system = EmailAutomationSystem(processor)

    # Process all sample emails
    results = []
    for email in sample_emails:
        logger.info(f"\nProcessing email {email['id']}...")
        result = automation_system.process_email(email)
        results.append(result)

    # Create a summary DataFrame
    df = pd.DataFrame(results)
    print("\nProcessing Summary:")
    print(df[["email_id", "success", "classification", "response_sent"]])

    # Save the results to a CSV file
    df.to_csv("email_processing_results.csv", index=False)

    return df


# Example usage:
if __name__ == "__main__":
    results_df = run_demonstration()
