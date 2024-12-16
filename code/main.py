import os
from pathlib import Path
import shutil
import pandas as pd
from data_processing import load_texts_from_file
from utils import visualise_aggregated_report, compile_aggregated_report, save_human_readable_report
from analysis import process_single_text
from data_processing import save_knowledgebase
import spacy
from concurrent.futures import ProcessPoolExecutor


def initialise_spacy_pipeline():
    """Initialise the SpaCy NLP pipeline."""
    print("[INFO] Initialising SpaCy pipeline...")
    nlp = spacy.load("de_core_news_lg")
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("coreferee")
    print("[INFO] SpaCy pipeline initialised.")
    return nlp


def process_single_file_multiprocessing(file_path, temp_dir, max_articles=None):
    """
    Process a single JSON file in a separate process.

    Args:
        file_path (str): Path to the JSON file.
        temp_dir (Path): Path to the temporary directory for storing results.
        max_articles (int, optional): Maximum number of articles to process. Defaults to None.
    """
    try:
        print(f"[INFO] Processing file: {file_path}")

        # Initialise the full SpaCy pipeline inside the process
        nlp = initialise_spacy_pipeline()
        
        stopwords = nlp.Defaults.stop_words
                
        print(f"[INFO] SpaCy pipeline initialised for {file_path}")

        raw_texts = load_texts_from_file(file_path, limit=max_articles)
        for article_id, text in raw_texts.items():
            try:
                doc = nlp(text)  # Process the text with the fully-initialised pipeline
                knowledgebase = process_single_text(doc , stopwords)

                if not knowledgebase.empty:
                    save_knowledgebase(knowledgebase, temp_dir, article_id)
            except Exception as e:
                print(f"[ERROR] Failed to process Article ID {article_id}: {e}")
    except Exception as e:
        print(f"[ERROR] Error processing file {file_path}: {e}")


def process_directory(directory, temp_dir, max_articles=None, max_workers=4):
    """
    Process all JSON files in a directory in parallel using multiprocessing.

    Args:
        directory (str): Path to the directory containing JSON files.
        temp_dir (Path): Path to the temporary directory for storing results.
        max_articles (int, optional): Maximum number of articles to process. Defaults to None.
        max_workers (int, optional): Number of parallel workers. Defaults to 4.

    Returns:
        None
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    json_files = [os.path.join(directory, file_name) for file_name in os.listdir(directory) if file_name.endswith(".json")]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_file_multiprocessing, file_path, temp_dir, max_articles)
            for file_path in json_files
        ]

        for future in futures:
            try:
                future.result()  # Wait for completion and catch any exceptions
            except Exception as e:
                print(f"[ERROR] Error during processing: {e}")


def analyse_and_visualise_year(directory, temp_dir, results_dir, year, max_articles=None):
    print(f"[INFO] Starting analysis for {year}...")
    process_directory(directory, temp_dir, max_articles)
    
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load and combine all knowledgebases
    print(f"[INFO] Loading processed knowledgebases from {temp_dir}...")
    all_knowledgebases = []
    for file_path in temp_dir.glob("knowledgebase_*.pkl"):
        try:
            print(f"[INFO] Loading {file_path.name}...")
            knowledgebase = pd.read_pickle(file_path)
            if not knowledgebase.empty:
                all_knowledgebases.append(knowledgebase)
                print(f"[INFO] Loaded knowledgebase with {len(knowledgebase)} entries.")
            else:
                print(f"[WARNING] {file_path.name} is empty, skipping.")
        except Exception as e:
            print(f"[ERROR] Failed to load knowledgebase from {file_path}: {e}")

    if all_knowledgebases:
        print(f"[INFO] Compiling aggregated report for {year}...")
        combined_knowledgebase = pd.concat(all_knowledgebases, ignore_index=True)
        aggregated_report, individual_values = compile_aggregated_report(all_knowledgebases)
        print(f"[INFO] Aggregated report for {year} generated.")

        # Save the aggregated knowledgebase
        aggregated_knowledgebase_path = results_dir / f"aggregated_knowledgebase_{year}.pkl"
        combined_knowledgebase.to_pickle(aggregated_knowledgebase_path)
        print(f"[INFO] Aggregated knowledgebase saved to {aggregated_knowledgebase_path}.")

        # Save the human-readable report
        save_human_readable_report(aggregated_report, results_dir, year, temp_dir)

        # Save visualisations
        print(f"[INFO] Saving visualisations for {year} to {results_dir}...")
        visualise_aggregated_report(individual_values, results_dir)
        print(f"[INFO] Visualisations saved for {year}.")
    else:
        print(f"[INFO] No knowledgebases processed for {year}.")

    # Clean up temporary directory
    # print(f"[INFO] Cleaning up temporary files for {year}...")
    # shutil.rmtree(temp_dir, ignore_errors=True)
    # print(f"[INFO] Temporary files cleaned up for {year}.")


def main():
    """
    Main function to process and analyse data for 1993 and 2023.
    """
    directories = {
        "2023": "data/2023",
        "1993": "data/1993"
    }

    TEMP_DIR = Path("tmp")
    RESULTS_DIR = Path("results")

    for year, directory in directories.items():
        print(f"\n[INFO] Starting processing for year {year}...\n")
        year_temp_dir = TEMP_DIR / year
        year_results_dir = RESULTS_DIR / year
        
        print(f"[INFO] Analysing data in {directory}")
        analyse_and_visualise_year(
            directory=directory,
            temp_dir=year_temp_dir,
            results_dir=year_results_dir,
            year=year,
            max_articles=None  # Set to None to process all articles
        )


if __name__ == "__main__":
    main()
