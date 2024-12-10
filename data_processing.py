import os
import json


def validate_chunk_size(chunk_size):
    """Ensure chunk size is a positive integer."""
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")


def load_texts_from_directory(directory, chunk_size=100):
    """Loads texts from JSON files in chunks."""
    validate_chunk_size(chunk_size)
    texts = []
    for file_name in os.listdir(directory):
        if not file_name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, file_name), "r", encoding="utf-8") as file:
                data = json.load(file)
                texts.extend(
                    article_data["text"]["text"]
                    for article_data in data.values()
                    if article_data.get("text", {}).get("text", "").strip()
                )
        except Exception as e:
            print(f"[ERROR] Skipping file {file_name}: {e}")
    return [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
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


