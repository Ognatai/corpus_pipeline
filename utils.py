import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
from pathlib import Path

# Generic terms for women in German
generic_woman = {
    "frau", "frauen", "mutter", "mütter", "mama", "oma", "großmutter", "dame", "damen"
}

# Generic terms for men in German
generic_man = {
    "mann", "männer", "vater", "väter", "papa", "opa", "großvater", "herr", "herren"
}

# Combine generic terms
generic_terms = generic_woman | generic_man

# Define lists of masculine- and feminine-coded words
masculine_coded_words = {
    "abenteuer", "aggressiv", "ambition", "analytisch", "aufgabenorientiert", "autark", "autoritär", "autonom",
    "beharr", "besieg", "bestimmt", "direkt", "domin", "durchsetz", "ehrgeiz", "eigenständig", "einfluss", "einflussreich",
    "energisch", "entscheid", "entschlossen", "erfolgsorientiert", "führ", "gewinn", "hartnäckig", "herausfordern",
    "hierarch", "kompetitiv", "konkurrenz", "kräftig", "kraft", "leisten", "leistungsfähig", "leistungsorientiert", "leit",
    "lenken", "mutig", "offensiv", "persisten", "rational", "risiko", "selbstbewusst", "selbstsicher", "selbstständig",
    "selbstvertrauen", "stark", "stärke", "stolz", "überlegen", "unabhängig", "wettbewerb", "wetteifer", "wettkampf",
    "wettstreit", "willens", "zielorientiert", "zielsicher", "zielstrebig"
}

feminine_coded_words = {
    "angenehm", "aufrichtig", "beraten", "bescheiden", "betreu", "beziehung", "commit", "dankbar", "ehrlich", "einfühl",
    "emotion", "empath", "engag", "familie", "fleiß", "förder", "freundlich", "freundschaft", "fürsorg", "gefühl",
    "gemeinsam", "gemeinschaft", "gruppe", "harmon", "helfen", "herzlich", "hilf", "höflich", "interpers", "kollabor",
    "kollegial", "kooper", "kümmern", "liebenswürdig", "loyal", "miteinander", "mitfühl", "mitgefühl", "nett",
    "partnerschaftlich", "pflege", "rücksicht", "sensibel", "sozial", "team", "treu", "umgänglich", "umsichtig",
    "uneigennützig", "unterstütz", "verantwortung", "verbunden", "verein", "verlässlich", "verständnis", "vertrauen",
    "wertschätz", "zugehörig", "zusammen", "zuverlässig", "zwischenmensch"
}


def get_actors(text):
    """
    Extract named entities and dependency-based actors from the text.

    Args:
        text (spacy.Doc): Preprocessed text.

    Returns:
        dict: A dictionary mapping actor names to their tokens.
    """
    actor_dict = {}

    for token in text:
        # Add compound entities linked to a PERSON
        if token.dep_ == "compound" and token.head.ent_type_ == "PER":
            actor_dict.setdefault(token.head.text, []).append(token)

        # Add individual PERSON entities
        elif token.ent_type_ == "PER" and "'" not in token.text:
            actor_dict.setdefault(token.text, []).append(token)

    return actor_dict


def get_generic_names(text, actor_dict):
    """
    Add generic names (e.g., "frau", "herr") to the actor dictionary.

    Args:
        text (spacy.Doc): Preprocessed text.
        actor_dict (dict): Existing actor dictionary.

    Returns:
        dict: Updated actor dictionary with generic names included.
    """
    
    for token in text:
        token_lower = token.text.lower()
        if token_lower in generic_terms:
            actor_dict.setdefault(token_lower, []).append(token)

    return actor_dict


def combine_names(actor_dict):
    """
    Combines similar actor names in the actor dictionary by merging their nominations.

    Args:
        actor_dict (dict): A dictionary with actor names as keys and their tokens as values.

    Returns:
        dict: A modified dictionary with combined actor names.
    """
    flagged_keys = {key for key in actor_dict if any(key in second_key for second_key in actor_dict if key != second_key)}
    for key in flagged_keys:
        for second_key in actor_dict:
            if key in second_key:
                actor_dict[second_key].extend(actor_dict[key])
    for key in flagged_keys:
        actor_dict.pop(key, None)
    
    return actor_dict


def get_pronouns_gender(knowledgebase, text):
    """
    Assume gender to actors based on German pronouns and generic terms.

    Args:
        knowledgebase (pd.DataFrame): DataFrame with actor nominations.
        text: A spaCy-parsed document.

    Returns:
        pd.DataFrame: Updated knowledgebase with gender assignments.
    """
    woman_pronouns = {"sie", "ihr", "ihre", "ihren", "ihrem", "ihres"}
    man_pronouns = {"er", "sein", "seine", "seinen", "seinem", "seines"}

    knowledgebase["pronoun"] = [[] for _ in range(len(knowledgebase))]
    knowledgebase["gender"] = "unknown"
    
    # Extract pronouns and save them to actors
    for token in text:
        if token.pos_ == "PRON" and hasattr(text._, "coref_chains"):
            resolved_actor = text._.coref_chains.resolve(token)
            if resolved_actor and len(resolved_actor) == 1:
                actor_name = resolved_actor[0].text
                for index, nomination in knowledgebase["nomination"].items():
                    if any(actor_name == nom.text for nom in nomination):
                        knowledgebase.at[index, "pronoun"].append(token)

    # Assume gender based on majority pronoun
    majority_threshold = 0.7
    for index, pronouns in knowledgebase["pronoun"].items():
        if pronouns:
            woman_count = sum(1 for p in pronouns if p.text.lower() in woman_pronouns)
            man_count = sum(1 for p in pronouns if p.text.lower() in man_pronouns)

            if woman_count / len(pronouns) >= majority_threshold:
                knowledgebase.at[index, "gender"] = "woman"
            elif man_count / len(pronouns) >= majority_threshold:
                knowledgebase.at[index, "gender"] = "man"

    for index in knowledgebase.index:
        if knowledgebase.at[index, 'gender'] == 'unknown':
            if index in generic_woman:
                knowledgebase.at[index, 'gender'] = 'woman'
            elif index in generic_man:
                knowledgebase.at[index, 'gender'] = 'man'

    return knowledgebase


def build_knowledgebase_nomination(text):
    """
    Constructs a knowledgebase of actors mentioned in the text,
    including named entities, generic terms, and combined names.

    Args:
        text (spacy.Doc): Preprocessed text with SpaCy pipeline.

    Returns:
        pd.DataFrame: A DataFrame containing actor names and their tokens.
    """
    # Step 1: Extract named entities and dependency-based actors and combine entities that belong together
    actors = combine_names(get_actors(text))  # Capture both named entities and dependency relations

    # Step 2: Add generic names like "frau" or "herr"
    actors = get_generic_names(text, actors)

    # Step 3: Create a DataFrame for the knowledgebase
    knowledgebase = pd.Series(actors).to_frame()
    knowledgebase.rename(columns={0: "nomination"}, inplace=True)
    
    # Step 4: Add pronouns and gender to the knowledgebase
    knowledgebase = get_pronouns_gender(knowledgebase, text)

    return knowledgebase


def calculate_pmi(word_freq, actor_word_freq, total_words, total_actor_words):
    """
    Calculates PMI for words associated with each actor.

    Args:
        word_freq (dict): Global word frequencies.
        actor_word_freq (dict): Actor-specific word frequencies.
        total_words (int): Total number of words in the corpus.
        total_actor_words (int): Total number of words associated with the actor.

    Returns:
        dict: PMI scores for each word.
    """
    pmi_scores = {}
    for word, freq in actor_word_freq.items():
        p_word_and_actor = freq / total_actor_words
        p_word = word_freq[word] / total_words
        p_actor = total_actor_words / total_words

        if p_word_and_actor > 0 and p_word > 0 and p_actor > 0:
            pmi_scores[word] = math.log2(p_word_and_actor / (p_word * p_actor))
    return pmi_scores


def visualise_boxplot(data, title, xlabel, output_file, exclude_outliers=True):
    """
    Create and save a boxplot for the given data, optionally excluding the top 10 outliers.

    Args:
        data (dict): Data to plot. Keys are groups, and values are lists of values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        output_file (str): Path to save the plot.
        exclude_outliers (bool): Whether to exclude the top 10 outliers for each group.
    """
    def remove_outliers(values):
        """
        Helper function to remove the top 10 outliers from a list of data.
        """
        # Filter out None values
        filtered_values = [v for v in values if v is not None]
        if len(filtered_values) > 20:  # Only remove outliers if sufficient data is available
            return sorted(filtered_values)[:-10]
        return filtered_values

    # Process data based on outlier exclusion setting
    if exclude_outliers:
        filtered_data = {key: remove_outliers(values) for key, values in data.items()}
    else:
        filtered_data = {key: [v for v in values if v is not None] for key, values in data.items()}

    # Remove empty groups
    filtered_data = {key: values for key, values in filtered_data.items() if values}

    if not filtered_data:
        print(f"[WARNING] All groups are empty after filtering; skipping plot: {title}")
        return

    # Prepare data for seaborn
    plt.figure(figsize=(12, 8))
    df = pd.DataFrame({key: pd.Series(values) for key, values in filtered_data.items()}).melt(var_name="Group", value_name="Values")

    sns.boxplot(data=df, x="Values", y="Group", orient="h", showfliers=exclude_outliers)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"[INFO] Saved boxplot to {output_file}.")


def visualise_aggregated_report(individual_values, output_dir):
    """
    Generate visualisations for all individual values in the aggregated report.

    Args:
        individual_values (dict): Dictionary of individual metrics.
        output_dir (str): Directory to save the visualisations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Gender and sentiment-related metrics
    gender_metrics = {
        "Gender Distribution": {
            "total":individual_values["total_actors"],
            "woman": individual_values["gender_distribution_woman"],
            "man": individual_values["gender_distribution_man"],
            "unknown": individual_values["gender_distribution_unknown"],
        },
        "Mentions by Gender": {
            "total":individual_values["total_mentions"],
            "woman": individual_values["mentions_gender_distribution_woman"],
            "man": individual_values["mentions_gender_distribution_man"],
            "unknown": individual_values["mentions_gender_distribution_unknown"],
        },
        "Sentiment by Gender": {
            "total":individual_values["average_sentiment_all"],
            "woman": individual_values["sentiment_by_gender_woman"],
            "man": individual_values["sentiment_by_gender_man"],
            "unknown": individual_values["sentiment_by_gender_unknown"],
        }
    }

    for metric, data in gender_metrics.items():
        output_file = os.path.join(output_dir, f"{metric.replace(' ', '_')}_boxplot.png")
        visualise_boxplot(
            data, title=f"{metric} Boxplot", xlabel=metric, output_file=output_file, exclude_outliers=True
        )

    # Binary metrics (convert lists into dictionaries for compatibility)
    binary_metrics = {
        "Contains Majority Gender Neutral": {"True": [v for v in individual_values["contains_majority_gender_neutral"] if v],
                                             "False": [v for v in individual_values["contains_majority_gender_neutral"] if not v]},
        "Generic Masculine": {"True": [v for v in individual_values["generic_masculine"] if v],
                              "False": [v for v in individual_values["generic_masculine"] if not v]},
    }

    for metric, data in binary_metrics.items():
        output_file = os.path.join(output_dir, f"{metric.replace(' ', '_')}_boxplot.png")
        visualise_boxplot(
            data, title=f"{metric} Boxplot", xlabel=metric, output_file=output_file, exclude_outliers=False
        )

    print(f"[INFO] Boxplots saved in {output_dir}")

            
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

