import pandas as pd
import math
from collections import Counter
import re
from transformers import pipeline

# Load sentiment analysis pipeline globally for efficiency
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

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
    Determines if the majority of potentially gendered words in a text use gender-neutral forms.

    Args:
        doc (spacy.tokens.Doc): The SpaCy document to analyse.

    Returns:
        bool: True if the majority of potentially gendered words are gender-neutral, False otherwise.
    """
    # Convert SpaCy document to string
    text = doc.text
    
    # Define regex pattern for gender-neutral forms
    gender_neutral_pattern = r".*[a-zäöüß][IR][a-zäöüß].*|.*[:*/_].*"
    
    # Split text into words and extract capitalised words
    words = text.split()
    capitalised_words = [word for word in words if word[0].isupper()]
    
    if not capitalised_words:
        return False  # If no capitalised words, default to False
    
    # Count gender-neutral forms
    gender_neutral_count = sum(1 for word in capitalised_words if re.match(gender_neutral_pattern, word))
    
    # Determine if the majority of capitalised words are gender-neutral
    return gender_neutral_count > (len(capitalised_words) / 2)
    
    
def detect_generic_masculine(doc):
    """
    Determines if only masculine forms of potentially gendered words are used in a SpaCy document.

    Args:
        doc (spacy.tokens.Doc): The SpaCy document to analyse.

    Returns:
        bool: True if only masculine forms are used, False otherwise.
    """
    # Convert SpaCy document to string
    text = doc.text
    
    # Define regex pattern for gender-neutral forms
    gender_neutral_pattern = r".*[a-zäöüß][IR][a-zäöüß].*|.*[:*/_].*"
    
    # Split text into words and extract capitalised words
    words = text.split()
    capitalised_words = [word for word in words if word[0].isupper()]
    
    if not capitalised_words:
        return False  # If no capitalised words, default to False
    
    # Count gender-neutral forms
    gender_neutral_count = sum(1 for word in capitalised_words if re.match(gender_neutral_pattern, word))

    # If there are no gender-neutral words but there are gendered candidates, it’s generic masculine
    return gender_neutral_count == 0 and len(capitalised_words) > 0


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


def compute_pmi(row, text, stopwords):
    """
    Compute PMI scores for a given actor's predications.

    Args:
        row (pd.Series): Actor data row.
        text (spacy.Doc): Entire processed text.
        stopwords (list): list of German stopwords.

    Returns:
        dict: PMI scores for each adjective in predications.
    """
    predication_words = [
        word.lower()
        for sentence in row["predication"]
        for word in sentence.split()
        if word.isalpha() and word.lower() not in stopwords
    ]
    actor_word_freq = Counter(predication_words)
    total_actor_words = sum(actor_word_freq.values())

    text_words = [
        tok.text.lower() 
        for tok in text 
        if tok.pos_ == "ADJ" and tok.is_alpha and tok.text.lower() not in stopwords
    ]
    word_freq = Counter(text_words)
    total_words = len(text_words)

    # Avoid division by zero
    if total_actor_words == 0 or total_words == 0:
        return {}

    return calculate_pmi(word_freq, actor_word_freq, total_words, total_actor_words)


def process_single_text(doc, stopwords):
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
                lambda row: compute_pmi(row, doc, stopwords), axis=1
        )
        
        knowledgebase["contains_majority_gender_neutral"] = detect_gender_neutral_language(doc)
        knowledgebase["generic_masculine"] = detect_generic_masculine(doc)
        
        knowledgebase["sentiment"] = knowledgebase["predication"].apply(detect_sentiment)

        return knowledgebase
    except Exception as e:
        print(f"[ERROR] process_single_text: {e}")
        return pd.DataFrame()


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
    flagged_keys = {key for key in actor_dict if
                    any(key in second_key for second_key in actor_dict if key != second_key)}
    for key in flagged_keys:
        for second_key in actor_dict:
            if key in second_key:
                actor_dict[second_key].extend(actor_dict[key])
    for key in flagged_keys:
        actor_dict.pop(key, None)

    return actor_dict


def get_pronouns_main(knowledgebase, text):
    """
    Get pronouns that are mainly used for an actor.

    Args:
        knowledgebase (pd.DataFrame): DataFrame with actor nominations.
        text: A spaCy-parsed document.

    Returns:
        pd.DataFrame: Updated knowledgebase with pronoun assignments.
    """
    she_her_pronouns = {"sie", "ihr", "ihre", "ihren", "ihrem", "ihres"}
    he_him_pronouns = {"er", "sein", "seine", "seinen", "seinem", "seines"}

    knowledgebase["pronoun"] = [[] for _ in range(len(knowledgebase))]
    knowledgebase["main_pronoun"] = " "

    # Extract pronouns and save them to actors
    for token in text:
        if token.pos_ == "PRON" and hasattr(text._, "coref_chains"):
            resolved_actor = text._.coref_chains.resolve(token)
            if resolved_actor and len(resolved_actor) == 1:
                actor_name = resolved_actor[0].text
                for index, nomination in knowledgebase["nomination"].items():
                    if any(actor_name == nom.text for nom in nomination):
                        knowledgebase.at[index, "pronoun"].append(token)

    # Assume main pronouns based on majority pronoun
    majority_threshold = 0.7
    for index, pronouns in knowledgebase["pronoun"].items():
        if pronouns:
            she_her_count = sum(1 for p in pronouns if p.text.lower() in she_her_pronouns)
            he_him_count = sum(1 for p in pronouns if p.text.lower() in he_him_pronouns)

            if she_her_count / len(pronouns) >= majority_threshold:
                knowledgebase.at[index, "main_pronoun"] = "she_her"
            elif he_him_count / len(pronouns) >= majority_threshold:
                knowledgebase.at[index, "main_pronoun"] = "he_him"

    # Drop entries without main pronoun
    knowledgebase.drop(knowledgebase[knowledgebase["main_pronoun"].str.strip() == ""].index, inplace=True)
        
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

    # Step 4: Add pronouns and main used pronouns to the knowledgebase and dropp all entries without main pronouns
    knowledgebase = get_pronouns_main(knowledgebase, text)

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

