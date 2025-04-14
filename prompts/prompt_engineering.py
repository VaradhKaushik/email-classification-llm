import dspy
from dspy import BootstrapFewShot
import os
import logging
import json
import random
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Callable, Literal, Optional

# --- Configuration ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Adjust path relative to this script's location (prompts/)
# Assumes .env is in the parent directory and data is in ../data/
try:
    # Load .env file from the specified relative path
    dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logger.info(f"Loaded environment variables from: {dotenv_path}")
    else:
        logger.warning(f".env file not found at {dotenv_path}. Relying on environment variables.")

except Exception as e:
    logger.error(f"Error loading .env file: {e}")


# Data and Output Paths (relative to project root for consistency if run from root)
# Or adjust if you always run this script directly from the prompts folder
DATA_FILE_PATH = "../data/labeled_emails.json" # Adjust if your labeled data is elsewhere
OUTPUT_CSV_PATH = "classification_prompt_comparison.csv" # Saves in the prompts folder

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = 'openai/gpt-4o' # Or your preferred model

# Classification Categories
VALID_CATEGORIES = {
    "complaint",
    "inquiry",
    "feedback",
    "support_request",
    "other"
}
EmailCategory = Literal[tuple(VALID_CATEGORIES)]

# Optimization Settings
TRAIN_SPLIT_RATIO = 0.7 # 70% for training optimizer, 30% for dev/evaluation
RANDOM_SEED = 42
MAX_BOOTSTRAPPED_DEMOS = 3 # Number of few-shot examples DSPy optimizer will try to create

# --- DSPy Signature Definition ---
class ClassifyEmailSignature(dspy.Signature):
    """Classify the email's primary category based on its subject and body.

    Focus on the main intent: is the user complaining, asking a question, providing feedback,
    requesting technical help, or something else?
    """
    email_subject: str = dspy.InputField(desc="The subject line of the email.")
    email_body: str = dspy.InputField(desc="The main content/body of the email.")
    category: EmailCategory = dspy.OutputField(
        desc=f"The classification category. Must be one of: {', '.join(VALID_CATEGORIES)}"
    )

# --- Data Loading and Preparation ---
def load_labeled_data(file_path: str = DATA_FILE_PATH) -> List[Dict]:
    """Loads labeled email data from a JSON file."""
    try:
        # Adjust path if running script directly from prompts/
        actual_path = file_path
        if not os.path.exists(actual_path) and not os.path.isabs(actual_path):
             script_dir = os.path.dirname(__file__)
             actual_path = os.path.join(script_dir, file_path) # Try relative to script

        if not os.path.exists(actual_path):
             logger.error(f"Labeled data file not found at attempted path: {actual_path}")
             return []

        with open(actual_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} labeled emails from {actual_path}")
        # Basic validation
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            logger.error("Invalid data format. Expected a list of dictionaries.")
            return []
        # Ensure required fields are present (adjust field names if needed)
        valid_data = [
            item for item in data
            if 'subject' in item and 'body' in item and 'category' in item and item['category'] in VALID_CATEGORIES
        ]
        if len(valid_data) < len(data):
            logger.warning(f"Filtered out {len(data) - len(valid_data)} items due to missing fields or invalid category.")
        return valid_data
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred loading labeled data: {e}")
        return []

def prepare_datasets(
    labeled_data: List[Dict],
    train_split_ratio: float = TRAIN_SPLIT_RATIO,
    seed: int = RANDOM_SEED
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Converts data to dspy.Example objects and splits into train/dev sets."""
    if not labeled_data:
        return [], []

    random.seed(seed)
    random.shuffle(labeled_data)

    split_index = int(len(labeled_data) * train_split_ratio)
    train_data_raw = labeled_data[:split_index]
    dev_data_raw = labeled_data[split_index:]

    # Convert to dspy.Example, matching signature fields
    trainset = [
        dspy.Example(
            email_subject=item['subject'],
            email_body=item['body'],
            category=item['category']
        ).with_inputs("email_subject", "email_body") # Specify inputs clearly
        for item in train_data_raw
    ]
    devset = [
        dspy.Example(
            email_subject=item['subject'],
            email_body=item['body'],
            category=item['category']
        ).with_inputs("email_subject", "email_body")
        for item in dev_data_raw
    ]

    logger.info(f"Data split: {len(trainset)} training examples, {len(devset)} development examples.")
    return trainset, devset

# --- LLM Configuration ---
def configure_llm():
    """Configures the DSPy LLM."""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError("Missing OpenAI API Key")
    try:
        llm = dspy.LM(model=LLM_MODEL_NAME, api_key=OPENAI_API_KEY, max_tokens=100) # Lower max_tokens for classification
        dspy.settings.configure(lm=llm)
        logger.info(f"DSPy configured with LLM: {LLM_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to configure DSPy LLM: {e}")
        raise

# --- Evaluation Metric ---
def classification_accuracy_metric(gold: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """Checks if predicted category matches the gold label (case-insensitive)."""
    predicted_category = getattr(prediction, 'category', None)
    expected_category = getattr(gold, 'category', None)
    return predicted_category is not None and \
           expected_category is not None and \
           str(predicted_category).strip().lower() == str(expected_category).strip().lower()

# --- Optimization Function ---
def optimize_classification_prompt(
    trainset: List[dspy.Example],
    devset: List[dspy.Example], # Keep devset for potential future use or different optimizers
    metric: Callable = classification_accuracy_metric,
    max_demos: int = MAX_BOOTSTRAPPED_DEMOS
) -> Optional[dspy.Module]:
    """Optimizes the classification prompt using BootstrapFewShot."""
    if not trainset: # BootstrapFewShot only strictly needs trainset
        logger.error("Training set is required for BootstrapFewShot optimization.")
        return None

    logger.info(f"Starting prompt optimization with BootstrapFewShot (max_demos={max_demos})...")

    # Configure the optimizer
    config = dict(max_bootstrapped_demos=max_demos)
    # Note: max_labeled_demos defaults based on max_bootstrapped_demos if not set
    optimizer = BootstrapFewShot(metric=metric, **config)

    # Define the base program to optimize (a simple Predict module)
    base_classifier = dspy.Predict(ClassifyEmailSignature)

    try:
        # Run the optimization using ONLY trainset for BootstrapFewShot
        optimized_classifier = optimizer.compile(
            student=base_classifier,
            trainset=trainset
        )
        logger.info("Prompt optimization finished successfully.")

        # --- BEGIN NEW: Log selected demos ---
        try:
            # Iterate through predictors in the compiled module
            demos_found = []
            for predictor in optimized_classifier.predictors():
                if hasattr(predictor, 'demos') and predictor.demos:
                    demos_found.extend(predictor.demos)

            if demos_found:
                logger.info(f"Optimizer selected {len(demos_found)} few-shot examples (demos):")
                # Log limited info to keep console clean, full demo object is complex
                for i, demo in enumerate(demos_found):
                    # Extract relevant fields - adjust keys based on your dspy.Example structure
                    subj = demo.get('email_subject', 'N/A')
                    cat = demo.get('category', 'N/A')
                    logger.info(f"  Demo {i+1}: Input Subject='{subj[:50]}...', Output Category='{cat}'")
            else:
                 logger.info("Optimizer did not store few-shot demos in the expected location.")

        except Exception as inspect_err:
             logger.warning(f"Could not inspect generated demos on the optimized program: {inspect_err}")
        # --- END NEW ---

        return optimized_classifier
    except Exception as e:
        logger.error(f"Error during DSPy optimization: {e}", exc_info=True)
        return None


# --- Evaluation and Comparison ---
def evaluate_and_compare(
    devset: List[dspy.Example],
    baseline_program: dspy.Module,
    optimized_program: Optional[dspy.Module],
    output_csv_path: str = OUTPUT_CSV_PATH
):
    """Evaluates baseline and optimized programs, saves comparison WITHOUT prompts to CSV.
       Relies on verbose logging triggered by inspect_history=True to show prompts.
    """
    results = []
    baseline_correct = 0
    optimized_correct = 0
    total_examples = len(devset)

    logger.info(f"Evaluating baseline vs optimized programs on {total_examples} examples...")
    logger.info("Prompts will be shown in verbose logs below (search for 'System message:' / 'User message:').")

    if not dspy.settings.lm:
        logger.error("DSPy LM not configured. Cannot proceed.")
        return

    for i, example in enumerate(devset):
        input_data = {"email_subject": example.email_subject, "email_body": example.email_body}
        expected_category = example.category

        # --- Baseline ---
        baseline_category = "ERROR_INIT"
        baseline_pred_obj = None
        baseline_correct_flag = False
        try:
            # Enable history tracking TO TRIGGER VERBOSE LOGS
            logger.info(f"\n--- [Example {i}] Processing BASELINE ---")
            with dspy.settings.context(inspect_history=True):
                baseline_pred_obj = baseline_program(**input_data)
            logger.info(f"--- [Example {i}] DONE Baseline Prediction ---") # Mark end

            # Process prediction results
            if baseline_pred_obj is not None:
                baseline_category = getattr(baseline_pred_obj, 'category', 'ERROR_PRED_ATTR')
                baseline_correct_flag = classification_accuracy_metric(example, baseline_pred_obj)
                if baseline_correct_flag:
                    baseline_correct += 1
            else:
                baseline_category = "ERROR_PRED_NONE"
                baseline_correct_flag = False

        except Exception as e:
            logger.error(f"Baseline processing failed for example {i}: {type(e).__name__} - {e}", exc_info=True)
            baseline_category = f"ERROR_EXC_{type(e).__name__}"
            baseline_correct_flag = False

        # --- Optimized ---
        optimized_category = "N/A"
        optimized_pred_obj = None
        optimized_correct_flag = False
        if optimized_program:
            try:
                logger.info(f"\n--- [Example {i}] Processing OPTIMIZED ---")
                # Enable history tracking TO TRIGGER VERBOSE LOGS
                with dspy.settings.context(inspect_history=True):
                    optimized_pred_obj = optimized_program(**input_data)
                logger.info(f"--- [Example {i}] DONE Optimized Prediction ---") # Mark end

                # Process prediction results
                if optimized_pred_obj is not None:
                    optimized_category = getattr(optimized_pred_obj, 'category', 'ERROR_PRED_ATTR')
                    optimized_correct_flag = classification_accuracy_metric(example, optimized_pred_obj)
                    if optimized_correct_flag:
                         optimized_correct += 1
                else:
                    optimized_category = "ERROR_PRED_NONE"
                    optimized_correct_flag = False

            except Exception as e:
                logger.error(f"Optimized processing failed for example {i}: {type(e).__name__} - {e}", exc_info=True)
                optimized_category = f"ERROR_EXC_{type(e).__name__}"
                optimized_correct_flag = False
        elif i == 0:
            logger.info("Skipping optimized predictions as optimization failed or was skipped.")

        # --- Store Results (NO Prompts) ---
        results.append({
            "Index": i,
            "Subject": example.email_subject,
            "Expected Category": expected_category,
            "Baseline Prediction": baseline_category,
            "Optimized Prediction": optimized_category,
            "Baseline Correct": baseline_correct_flag,
            "Optimized Correct": optimized_correct_flag,
            # REMOVED: "Baseline Prompt": baseline_prompt,
            # REMOVED: "Optimized Prompt": optimized_prompt
        })

    # --- Calculate Accuracies and Save (NO Prompts) ---
    baseline_accuracy = (baseline_correct / total_examples) * 100 if total_examples > 0 else 0
    optimized_accuracy = (optimized_correct / total_examples) * 100 if total_examples > 0 and optimized_program else 0

    logger.info(f"\nBaseline Accuracy: {baseline_accuracy:.2f}% ({baseline_correct}/{total_examples})")
    if optimized_program:
        logger.info(f"Optimized Accuracy: {optimized_accuracy:.2f}% ({optimized_correct}/{total_examples})")
    else:
        logger.info("Optimized Accuracy: N/A (Optimization skipped/failed)")

    # Save to CSV (NO Prompts)
    try:
        results_df = pd.DataFrame(results)
        # Define columns WITHOUT prompts
        output_columns = [
            "Index", "Subject", "Expected Category",
            "Baseline Prediction", "Optimized Prediction",
            "Baseline Correct", "Optimized Correct"
        ]
        # Ensure columns exist
        for col in output_columns:
            if col not in results_df.columns: results_df[col] = None

        if not os.path.isabs(output_csv_path):
             output_csv_path = os.path.join(os.path.dirname(__file__), output_csv_path)

        results_df[output_columns].to_csv(output_csv_path, index=False, encoding='utf-8')
        logger.info(f"Comparison results saved to: {output_csv_path}")
        print(f"\nComparison results saved to: {output_csv_path}")
        print("\nSample Results (Predictions Only):")
        print(results_df.head().to_string())
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {e}", exc_info=True)



# --- Main Execution ---
def main():
    """Main function to run the classification prompt optimization and evaluation."""
    logger.info("--- Starting Classification Prompt Engineering Script ---")

    # 1. Configure LLM
    try:
        configure_llm()
    except Exception as e:
        logger.error(f"Fatal Error: Could not configure LLM. Exiting. {e}")
        return

    # 2. Load Data
    labeled_data = load_labeled_data()
    if not labeled_data:
        logger.error("Fatal Error: No labeled data loaded. Exiting.")
        return

    # 3. Prepare Datasets
    trainset, devset = prepare_datasets(labeled_data)
    if not trainset or not devset:
        logger.error("Fatal Error: Could not create non-empty train/dev datasets. Exiting.")
        return

    # 4. Define Baseline Program
    baseline_classifier = dspy.Predict(ClassifyEmailSignature)
    logger.info("Defined baseline classifier (dspy.Predict).")

    # 5. Optimize Prompt
    optimized_classifier = optimize_classification_prompt(trainset, devset)
    if not optimized_classifier:
         logger.warning("Proceeding with evaluation using baseline only, as optimization failed.")
         # Fallback to baseline if optimization returns None
         # optimized_classifier = baseline_classifier # Or keep it None to mark as N/A

    # 6. Evaluate and Compare
    evaluate_and_compare(
        devset=devset,
        baseline_program=baseline_classifier,
        optimized_program=optimized_classifier # Will be None if optimization failed
    )

    logger.info("--- Classification Prompt Engineering Script Finished ---")

if __name__ == "__main__":
    main()
