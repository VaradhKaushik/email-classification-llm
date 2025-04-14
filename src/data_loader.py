import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .config import DATA_FILE_PATH, logger


def load_email_data(file_path: str = DATA_FILE_PATH) -> Optional[List[Dict]]:
    """
    Load email data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing email data.
        
    Returns:
        List[Dict]: A list of email dictionaries.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logger.info(f"Loaded {len(data)} emails from {file_path}")

        # Validation
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            logger.error(f"Invalid data format in {file_path}, expected a list of dictionaries.")
            return None

        required_keys = {"id", "subject", "body"}
        if not all(required_keys.issubset(item.keys()) for item in data):
            logger.error(f"Missing required keys in email data from {file_path}. Requiered keys: {required_keys}")
            return None
        
        return data
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}")
        return None

    except Exception as e:
        logger.error(f"Failed to load email data: {e}")
        return None

