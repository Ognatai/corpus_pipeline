import os
import json
from pathlib import Path
import pickle

    
def load_texts_from_file(file_path, limit=None):
    """
    Load raw texts from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing articles.
        limit (int, optional): Maximum number of texts to load. Defaults to None.

    Returns:
        dict: Dictionary of article IDs as keys and their raw texts as values.
    """
    try:
        # Ensure the file exists and is accessible
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Create a dictionary for raw texts
        raw_texts = {}
        count = 0
        for article_id, article_data in data.items():
            # Stop if the limit is reached
            if limit is not None and count >= limit:
                break

            # Extract and validate the text
            raw_text = article_data.get("text", {}).get("text", "").strip()
            if raw_text:
                raw_texts[article_id] = raw_text
                count += 1
            else:
                print(f"[DEBUG] Skipping empty or invalid text for Article ID: {article_id}")

        print(f"[INFO] Loaded {len(raw_texts)} articles from {file_path}.")
        return raw_texts

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON file {file_path}: {e}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while loading texts: {e}")

    # Return an empty dictionary if an error occurs
    return {}


def load_all_knowledgebases(temp_dir):
    """
    Load all knowledgebases from the specified directory.
    """
    knowledgebases = []
    for file_path in Path(temp_dir).glob("knowledgebase_*.pkl"):
        try:
            print(f"[INFO] Loading {file_path.name}...")
            knowledgebase = load_knowledgebase(file_path)
            if not knowledgebase.empty:
                knowledgebases.append(knowledgebase)
                print(f"[INFO] Knowledgebase loaded with {len(knowledgebase)} entries.")
            else:
                print(f"[WARNING] {file_path.name} is empty, skipping.")
        except Exception as e:
            print(f"[ERROR] Failed to load knowledgebase from {file_path}: {e}")
    return knowledgebases


def load_knowledgebase(file_path):
    """Load a knowledgebase from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            knowledgebase = pickle.load(f)
        print(f"[INFO] Knowledgebase loaded from {file_path}.")
        return knowledgebase
    except Exception as e:
        print(f"[ERROR] Failed to load knowledgebase from {file_path}: {e}")
        return pd.DataFrame()


def save_knowledgebase(knowledgebase, temp_dir, article_id):
    """Save individual knowledgebase to a pickle file."""
    if knowledgebase.empty:
        print(f"[DEBUG] Knowledgebase {article_id} is empty, skipping.")
        return

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / f"knowledgebase_{article_id}.pkl"

    try:
        # Convert tokens to text for saving
        knowledgebase = knowledgebase.copy()
        for column in ["nomination", "pronoun"]:
            if column in knowledgebase.columns:
                knowledgebase[column] = knowledgebase[column].apply(
                    lambda tokens: [token.text if hasattr(token, "text") else token for token in tokens]
                )

        with open(file_path, "wb") as f:
            pickle.dump(knowledgebase, f)
        print(f"[INFO] Knowledgebase saved to {file_path}.")
    except Exception as e:
        print(f"[ERROR] Failed to save knowledgebase for Article ID {article_id}: {e}")

