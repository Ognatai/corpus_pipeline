import pandas as pd
from collections import Counter
from collections import defaultdict
from utils import (
    masculine_coded_words,
    feminine_coded_words,
    calculate_pmi,
    build_knowledgebase_nomination
) 
import re
from transformers import pipeline

# Load sentiment analysis pipeline globally for efficiency
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")


def detect_sentiment(predications):
    """
    Detect the sentiment of each predication and calculate the average sentiment.

    Args:
        predications (list): List of predication sentences as strings.

    Returns:
        dict: Sentiment scores with "average_sentiment" and individual predication sentiments.
    """
    if not predications:
        return {"average_sentiment": 0.0, "individual_sentiments": []}

    sentiments = []
    for predication in predications:
        try:
            result = sentiment_pipeline(predication)
            # Convert sentiment label to numerical value
            sentiment_score = (
                result[0]["score"] if result[0]["label"] == "positive" else
                -result[0]["score"] if result[0]["label"] == "negative" else
                0.0  # Neutral case
            )
            sentiments.append(sentiment_score)
        except Exception as e:
            print(f"[ERROR] Sentiment analysis failed for predication '{predication}': {e}")
            sentiments.append(0.0)  # Default to neutral if sentiment fails

    # Calculate the average sentiment
    average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

    return {
        "average_sentiment": average_sentiment,
        "individual_sentiments": sentiments
    }


def detect_gender_neutral_language(doc):
    """
    Determines if the majority of potentially gendered words in a SpaCy document use gender-neutral forms.

    Args:
        doc (spacy.tokens.Doc): The SpaCy document to analyse.

    Returns:
        bool: True if the majority of potentially gendered words are gender-neutral, False otherwise.
    """
    # Define regex patterns for gender-neutral forms
    gender_neutral_pattern = r".*[a-zäöüß][IR][a-zäöüß].*|.*[:*/_].*"

    # Extract nouns and proper nouns as potentially gendered candidates
    gendered_candidates = [
        token.text for token in doc 
        if token.pos_ in {"NOUN", "PROPN"}  # Focus on nouns and proper nouns
    ]

    # Filter out words where capital "I" or "R" is simply sentence-initial or a proper noun
    filtered_candidates = [
        word for word in gendered_candidates
        if not (word.istitle() and word[0] in "IR")  # Exclude sentence-initial/proper noun "I" or "R"
        or re.search(gender_neutral_pattern, word)  # Include gender-neutral words like jedeR
    ]

    if not filtered_candidates:
        return False  # If no filtered candidates, default to False

    # Count gender-neutral forms
    gender_neutral_count = sum(
        1 for word in filtered_candidates if re.match(gender_neutral_pattern, word)
    )

    # Determine if the majority of gendered candidates are gender-neutral
    return gender_neutral_count > (len(filtered_candidates) / 2)
    
    
def detect_generic_masculine(doc):
    """
    Determines if only masculine forms of potentially gendered words are used in a SpaCy document.

    Args:
        doc (spacy.tokens.Doc): The SpaCy document to analyse.

    Returns:
        bool: True if only masculine forms are used, False otherwise.
    """
    # Define regex pattern for explicitly gendered forms
    gender_neutral_pattern = r".*[a-zäöüß][IR][a-zäöüß].*|.*[:*/_].*"
    
    # Extract nouns and proper nouns as potentially gendered candidates
    gendered_candidates = [
        token.text for token in doc 
        if token.pos_ in {"NOUN", "PROPN"}  # Focus on nouns and proper nouns
    ]

    # Count gender-neutral forms
    gender_neutral_count = sum(
        1 for word in gendered_candidates if re.match(gender_neutral_pattern, word)
    )

    # If there are no gender-neutral words but there are gendered candidates, it’s generic masculine
    return gender_neutral_count == 0 and len(gendered_candidates) > 0


def count_gendered_words(predications, gendered_words_list):
    """
    Count occurrences of gendered words in predications.

    Args:
        predications (list): List of predication sentences as strings.
        gendered_words_list (list): List of gendered words to search for.

    Returns:
        int: Total count of gendered words in the predications.
    """
    total_count = 0
    for sentence in predications:
        for word in gendered_words_list:
            total_count += sentence.lower().split().count(word)  # Count word occurrences
    return total_count


def extract_predications(nomination_tokens, pronoun_tokens, text):
    """
    Extract predication spans for an actor and return them as strings.

    Args:
        nomination_tokens (list): Tokens representing the actor's name.
        pronoun_tokens (list): Pronouns coreferenced with the actor.
        text (spacy.Doc): The full processed text.

    Returns:
        list: List of unique predication sentences as strings.
    """
    try:
        predications = []
        seen_sentences = set()

        for token in nomination_tokens + pronoun_tokens:
            sent = token.sent
            if sent not in seen_sentences:
                seen_sentences.add(sent)
                predications.append(sent.text)  # Store string representation

        return predications
    except RecursionError as re:
        print(f"[ERROR] Recursion error in extract_predications: {re}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error in extract_predications: {e}")
        return []


def compute_pmi(row, text):
    """
    Compute PMI scores for a given actor's predications.

    Args:
        row (pd.Series): Actor data row.
        text (spacy.Doc): Entire processed text.

    Returns:
        dict: PMI scores for each word in predications.
    """
    predication_words = [
        word.lower()
        for sentence in row["predication"]
        for word in sentence.split()
        if word.isalpha()
    ]
    actor_word_freq = Counter(predication_words)
    total_actor_words = sum(actor_word_freq.values())

    text_words = [token.text.lower() for token in text if token.is_alpha]
    word_freq = Counter(text_words)
    total_words = len(text_words)

    # Avoid division by zero
    if total_actor_words == 0 or total_words == 0:
        return {}

    return calculate_pmi(word_freq, actor_word_freq, total_words, total_actor_words)


def process_single_text(doc):
    """Processes a single text and builds the knowledgebase."""
    try:
        knowledgebase = build_knowledgebase_nomination(doc)
        if knowledgebase.empty:
            return pd.DataFrame()

        knowledgebase["predication"] = knowledgebase.apply(
            lambda row: extract_predications(row["nomination"], row["pronoun"], doc), axis=1
        )

        for column, word_list in {
            "feminine_coded_words": feminine_coded_words,
            "masculine_coded_words": masculine_coded_words,
        }.items():
            knowledgebase[column] = knowledgebase["predication"].apply(
                lambda preds: count_gendered_words(preds, word_list)
            )

        knowledgebase["mention_count"] = (
            knowledgebase["nomination"].apply(len)
            + knowledgebase["pronoun"].apply(len)
        )
        
        knowledgebase["pmi"] = knowledgebase.apply(
                lambda row: compute_pmi(row, doc), axis=1
        )
        
        knowledgebase["contains_majority_gender_neutral"] = detect_gender_neutral_language(doc)
        knowledgebase["generic_masculine"] = detect_generic_masculine(doc)
        
        knowledgebase["sentiment"] = knowledgebase["predication"].apply(detect_sentiment)

        return knowledgebase
    except Exception as e:
        print(f"[ERROR] process_single_text: {e}")
        return pd.DataFrame()

    
def generate_individual_report(knowledgebase):
    """
    Generate a report for a single knowledgebase.

    Args:
        knowledgebase (pd.DataFrame): Knowledgebase for a single chunk.

    Returns:
        dict: Aggregated metrics for the knowledgebase.
    """
    if knowledgebase.empty:
        return {}

    try:
        # claculate metrics
        total_actors = len(knowledgebase)
        gender_distribution = knowledgebase["gender"].value_counts().to_dict()
        total_mentions = knowledgebase["mention_count"].sum()
        mentions_gender_distribution = knowledgebase.groupby("gender")["mention_count"].sum().to_dict()
        total_feminine_coded_words = knowledgebase["feminine_coded_words"].sum()
        feminine_coded_words_gender_distribution = knowledgebase.groupby("gender")["feminine_coded_words"].sum().to_dict()
        total_masculine_coded_words = knowledgebase["masculine_coded_words"].sum()
        masculine_coded_words_gender_distribution = knowledgebase.groupby("gender")["masculine_coded_words"].sum().to_dict()
        top_pmi_words = Counter(
            word for pmi in knowledgebase["pmi"] for word in (pmi or {})
        ).most_common(10)
        contains_majority_gender_neutral = knowledgebase["contains_majority_gender_neutral"].sum()
        generic_masculine = knowledgebase["generic_masculine"].sum()
        average_sentiment_all = knowledgebase["sentiment"].apply(lambda x: x["average_sentiment"]).mean()
        sentiment_by_gender = {}
        for gender in ["woman", "man", "unknown"]:
            gender_data = knowledgebase[knowledgebase["gender"] == gender]
            sentiment_by_gender[gender] = (
                gender_data["sentiment"].apply(lambda x: x["average_sentiment"]).mean()
                if not gender_data.empty
                else None
            )
        
        report = {
            "total_actors": total_actors,
            "gender_distribution": gender_distribution,
            "total_mentions": total_mentions,
            "mentions_gender_distribution": mentions_gender_distribution,
            "total_feminine_coded_words": total_feminine_coded_words,
            "feminine_coded_words_gender_distribution": feminine_coded_words_gender_distribution,
            "total_masculine_coded_words": total_masculine_coded_words,
            "masculine_coded_words_gender_distribution": masculine_coded_words_gender_distribution,
            "top_pmi_words": top_pmi_words,
            "contains_majority_gender_neutral": contains_majority_gender_neutral,
            "generic_masculine": generic_masculine,
            "average_sentiment": average_sentiment_all,
            "sentiment_by_gender": sentiment_by_gender
        }
    except Exception as e:
        print(f"[ERROR] Error in generate_individual_report: {e}")
        return {}
    
    return report


def compile_aggregated_report(knowledgebases):
    """
    Compile an aggregated report from a list of knowledgebases.

    Args:
        knowledgebases (list): List of knowledgebase DataFrames.

    Returns:
        tuple: Aggregated report and dictionary with all individual report values.
    """
    # Initialise accumulated data
    accumulated_data = {
        "total_actors": 0,
        "gender_distribution": defaultdict(int),
        "total_mentions": 0,
        "mentions_gender_distribution": defaultdict(int),
        "total_feminine_coded_words": 0,
        "feminine_coded_words_gender_distribution": defaultdict(int),
        "total_masculine_coded_words": 0,
        "masculine_coded_words_gender_distribution": defaultdict(int),
        "contains_majority_gender_neutral": 0,
        "generic_masculine": 0,
        "pmi_words": Counter(),
        "average_sentiment_all": [],
        "sentiment_by_gender": {
            "woman": [],
            "man": [],
            "unknown": [],
        }
    }

    # Flattened individual values
    individual_values = { 
        "total_actors": [],
        "total_mentions": [],
        "total_feminine_coded_words": [],
        "total_masculine_coded_words": [],
        "contains_majority_gender_neutral": [],
        "generic_masculine": [],
        "gender_distribution_woman": [],
        "gender_distribution_man": [],
        "gender_distribution_unknown": [],
        "mentions_gender_distribution_woman": [],
        "mentions_gender_distribution_man": [],
        "mentions_gender_distribution_unknown": [],
        "feminine_coded_words_gender_distribution_woman": [],
        "feminine_coded_words_gender_distribution_man": [],
        "feminine_coded_words_gender_distribution_unknown": [],
        "masculine_coded_words_gender_distribution_woman": [],
        "masculine_coded_words_gender_distribution_man": [],
        "masculine_coded_words_gender_distribution_unknown": [],
        "average_sentiment_all": [],
        "sentiment_by_gender_woman": [],
        "sentiment_by_gender_man": [],
        "sentiment_by_gender_unknown": []
    }

    # Count documents with the flags set
    docs_with_majority_gender_neutral = 0
    docs_with_generic_masculine = 0

    for idx, kb in enumerate(knowledgebases):
        if kb.empty:
            print(f"[DEBUG] Knowledgebase {idx} is empty, skipping.")
            continue

        report = generate_individual_report(kb)
        if not report:
            print(f"[DEBUG] No report generated for knowledgebase {idx}, skipping.")
            continue

        # Count these flags per document
        if report["contains_majority_gender_neutral"]:
            docs_with_majority_gender_neutral += 1
        if report["generic_masculine"]:
            docs_with_generic_masculine += 1

        # Update individual values
        individual_values["total_actors"].append(report["total_actors"])
        individual_values["total_mentions"].append(report["total_mentions"])
        individual_values["total_feminine_coded_words"].append(report["total_feminine_coded_words"])
        individual_values["total_masculine_coded_words"].append(report["total_masculine_coded_words"])
        individual_values["average_sentiment_all"].append(report["average_sentiment"])
        individual_values["contains_majority_gender_neutral"].append(report["contains_majority_gender_neutral"])
        individual_values["generic_masculine"].append(report["generic_masculine"])

        for subkey in ["woman", "man", "unknown"]:
            individual_values[f"gender_distribution_{subkey}"].append(
                report["gender_distribution"].get(subkey, 0)
            )
            individual_values[f"mentions_gender_distribution_{subkey}"].append(
                report["mentions_gender_distribution"].get(subkey, 0)
            )
            individual_values[f"feminine_coded_words_gender_distribution_{subkey}"].append(
                report["feminine_coded_words_gender_distribution"].get(subkey, 0)
            )
            individual_values[f"masculine_coded_words_gender_distribution_{subkey}"].append(
                report["masculine_coded_words_gender_distribution"].get(subkey, 0)
            )
            individual_values[f"sentiment_by_gender_{subkey}"].append(
                report["sentiment_by_gender"].get(subkey, 0)
            )

        # Aggregate totals
        accumulated_data["total_actors"] += report["total_actors"]
        accumulated_data["contains_majority_gender_neutral"] += report["contains_majority_gender_neutral"]
        accumulated_data["generic_masculine"] += report["generic_masculine"]
        for key, value in report["gender_distribution"].items():
            accumulated_data["gender_distribution"][key] += value

        accumulated_data["total_mentions"] += report["total_mentions"]
        for key, value in report["mentions_gender_distribution"].items():
            accumulated_data["mentions_gender_distribution"][key] += value

        accumulated_data["total_feminine_coded_words"] += report["total_feminine_coded_words"]
        for key, value in report["feminine_coded_words_gender_distribution"].items():
            accumulated_data["feminine_coded_words_gender_distribution"][key] += value

        accumulated_data["total_masculine_coded_words"] += report["total_masculine_coded_words"]
        for key, value in report["masculine_coded_words_gender_distribution"].items():
            accumulated_data["masculine_coded_words_gender_distribution"][key] += value

        if isinstance(report["top_pmi_words"], list):
            accumulated_data["pmi_words"].update(dict(report["top_pmi_words"]))
            
        for gender in ["woman", "man", "unknown"]:
            if report["sentiment_by_gender"][gender] is not None:
                accumulated_data["sentiment_by_gender"][gender].append(report["sentiment_by_gender"][gender])

        accumulated_data["average_sentiment_all"].append(report["average_sentiment"])
        
    # Aggregate sentiment
    average_sentiment_all = (
        sum(accumulated_data["average_sentiment_all"]) / len(accumulated_data["average_sentiment_all"])
        if accumulated_data["average_sentiment_all"]
        else None
    )
    sentiment_by_gender = {
        gender: (sum(values) / len(values)) if values else None
        for gender, values in accumulated_data["sentiment_by_gender"].items()
    }

    # Convert accumulated data counters to dictionaries
    accumulated_data["gender_distribution"] = dict(accumulated_data["gender_distribution"])
    accumulated_data["mentions_gender_distribution"] = dict(accumulated_data["mentions_gender_distribution"])
    accumulated_data["feminine_coded_words_gender_distribution"] = dict(
        accumulated_data["feminine_coded_words_gender_distribution"]
    )
    accumulated_data["masculine_coded_words_gender_distribution"] = dict(
        accumulated_data["masculine_coded_words_gender_distribution"]
    )

    # Calculate averages and medians
    df = pd.DataFrame(individual_values)
    averages = df.mean(numeric_only=True).to_dict()
    medians = df.median(numeric_only=True).to_dict()

    # Prepare the aggregated report
    aggregated_report = {
        "total_actors": accumulated_data["total_actors"],
        "gender_distribution": accumulated_data["gender_distribution"],
        "total_mentions": accumulated_data["total_mentions"],
        "mentions_gender_distribution": accumulated_data["mentions_gender_distribution"],
        "total_feminine_coded_words": accumulated_data["total_feminine_coded_words"],
        "feminine_coded_words_gender_distribution": accumulated_data["feminine_coded_words_gender_distribution"],
        "total_masculine_coded_words": accumulated_data["total_masculine_coded_words"],
        "masculine_coded_words_gender_distribution": accumulated_data["masculine_coded_words_gender_distribution"],
        "contains_majority_gender_neutral": docs_with_majority_gender_neutral,
        "generic_masculine": docs_with_generic_masculine,
        "average_metrics": averages,
        "median_metrics": medians,
        "top_pmi_words": accumulated_data["pmi_words"].most_common(10),
        "average_sentiment_all": average_sentiment_all,
        "sentiment_by_gender": sentiment_by_gender
    }

    return aggregated_report, individual_values

