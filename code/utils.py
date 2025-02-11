import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import load_all_knowledgebases
from collections import Counter
from collections import defaultdict


def save_human_readable_report(aggregated_report, results_dir, year, temp_dir):
    """
    Save a human-readable report to a .txt file.

    Args:
        aggregated_report (dict): The aggregated report to format and save.
        results_dir (Path): Directory to save the report.
        year (int): The year associated with the report.
        temp_dir (Path): Directory containing individual knowledgebases.
    """
    report_path = results_dir / f"aggregated_report_{year}.txt"
    try:
        # Count total number of texts by counting the knowledge base files
        total_texts = len(list(temp_dir.glob("knowledgebase_*.pkl")))

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(f"Aggregated Report for {year}\n")
            file.write("=" * 60 + "\n\n")

            file.write(f"Total Texts: {total_texts}\n")
            file.write(f"Total Actors: {aggregated_report['total_actors']}\n")
            file.write(f"Pronoun Distribution: {aggregated_report['pronoun_distribution']}\n")
            file.write(f"Total Mentions: {aggregated_report['total_mentions']}\n")
            file.write(f"Mentions by Pronoun: {aggregated_report['mentions_pronoun_distribution']}\n\n")
            file.write("\nMean Metrics:\n")
            for metric, value in aggregated_report['average_metrics'].items():
                file.write(f"  {metric}: {value:.2f}\n")

            file.write("\nMedian Metrics:\n")
            for metric, value in aggregated_report['median_metrics'].items():
                file.write(f"  {metric}: {value:.2f}\n")
                
            file.write("\nTop PMI Adjectives Table:\n")
            file.write(f"{'All':<20} {'she/her':<20} {'he/him':<20}\n")
            file.write("-" * 80 + "\n")

            # Prepare lists of words and scores for each category
            top_all = aggregated_report["top_pmi_words"]
            top_she_her = aggregated_report["top_pmi_words_pronoun_distribution"].get("she_her", [])
            top_he_him = aggregated_report["top_pmi_words_pronoun_distribution"].get("he_him", [])

            # Ensure all lists have the same length
            max_length = max(len(top_all), len(top_she_her), len(top_he_him))
            top_all += [("", 0)] * (max_length - len(top_all))
            top_she_her += [("", 0)] * (max_length - len(top_she_her))
            top_he_him += [("", 0)] * (max_length - len(top_he_him))
            
            # Write rows for the table
            for i in range(max_length):
                all_word, all_score = top_all[i]
                she_her_word, she_her_score = top_she_her[i]
                he_him_word, he_him_score = top_he_him[i]

                file.write(
                    f"{all_word:<20} {she_her_word:<20} {he_him_word:<20}\n"
                )

        print(f"[INFO] Human-readable report saved to {report_path}")
    except FileNotFoundError:
        print(f"[ERROR] Directory for year {year} does not exist: {tmp_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save human-readable report: {e}")




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
        # calculate metrics
        total_actors = len(knowledgebase)
        pronoun_distribution = knowledgebase["main_pronoun"].value_counts().to_dict()
        total_mentions = knowledgebase["mention_count"].sum()
        mentions_pronoun_distribution = knowledgebase.groupby("main_pronoun")["mention_count"].sum().to_dict()
        total_feminine_coded_words = knowledgebase["feminine_coded_words"].sum()
        feminine_coded_words_pronoun_distribution = knowledgebase.groupby("main_pronoun")[
            "feminine_coded_words"].sum().to_dict()
        total_masculine_coded_words = knowledgebase["masculine_coded_words"].sum()
        masculine_coded_words_pronoun_distribution = knowledgebase.groupby("main_pronoun")[
            "masculine_coded_words"].sum().to_dict()
        top_pmi_words = Counter(
            word for pmi in knowledgebase["pmi"] for word in (pmi or {})
        ).most_common(10)
        contains_majority_gender_neutral = knowledgebase["contains_majority_gender_neutral"].sum()
        generic_masculine = knowledgebase["generic_masculine"].sum()
        average_sentiment_all = knowledgebase["sentiment"].apply(lambda x: x["average_sentiment"]).mean()
        sentiment_by_pronoun = {}
        for pronoun in ["she_her", "he_him"]:
            pronoun_data = knowledgebase[knowledgebase["main_pronoun"] == pronoun]
            sentiment_by_pronoun[pronoun] = (
                pronoun_data["sentiment"].apply(lambda x: x["average_sentiment"]).mean()
                if not pronoun_data.empty
                else None
            )
        top_pmi_words_pronoun_distribution = {}
        for pronoun in ["she_her", "he_him"]:
            pronoun_data = knowledgebase[knowledgebase["main_pronoun"] == pronoun]
            pmi_counter = Counter(
                word
                for pmi in pronoun_data["pmi"]
                for word in (pmi or {})
            )
            top_pmi_words_pronoun_distribution[pronoun] = pmi_counter.most_common(10)

        report = {
            "total_actors": total_actors,
            "pronoun_distribution": pronoun_distribution,
            "total_mentions": total_mentions,
            "mentions_pronoun_distribution": mentions_pronoun_distribution,
            "total_feminine_coded_words": total_feminine_coded_words,
            "feminine_coded_words_pronoun_distribution": feminine_coded_words_pronoun_distribution,
            "total_masculine_coded_words": total_masculine_coded_words,
            "masculine_coded_words_pronoun_distribution": masculine_coded_words_pronoun_distribution,
            "top_pmi_words": top_pmi_words,
            "top_pmi_words_pronoun_distribution": top_pmi_words_pronoun_distribution,
            "contains_majority_gender_neutral": contains_majority_gender_neutral,
            "generic_masculine": generic_masculine,
            "average_sentiment": average_sentiment_all,
            "sentiment_by_pronoun": sentiment_by_pronoun
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
        "pronoun_distribution": defaultdict(int),
        "total_mentions": 0,
        "mentions_pronoun_distribution": defaultdict(int),
        "total_feminine_coded_words": 0,
        "feminine_coded_words_pronoun_distribution": defaultdict(int),
        "total_masculine_coded_words": 0,
        "masculine_coded_words_pronoun_distribution": defaultdict(int),
        "contains_majority_gender_neutral": 0,
        "generic_masculine": 0,
        "pmi_words": Counter(),
        "top_pmi_words_pronoun_distribution":  {
            "she_her": Counter(),
            "he_him": Counter()
        },
        "average_sentiment_all": [],
        "sentiment_by_pronoun": {
            "she_her": [],
            "he_him": []
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
        "pronoun_distribution_she_her": [],
        "pronoun_distribution_he_him": [],
        "mentions_pronoun_distribution_she_her": [],
        "mentions_pronoun_distribution_he_him": [],
        "feminine_coded_words_pronoun_distribution_she_her": [],
        "feminine_coded_words_pronoun_distribution_he_him": [],
        "masculine_coded_words_pronoun_distribution_she_her": [],
        "masculine_coded_words_pronoun_distribution_he_him": [],
        "average_sentiment_all": [],
        "sentiment_by_pronoun_she_her": [],
        "sentiment_by_pronoun_he_him": []
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

        for subkey in ["she_her", "he_him"]:
            individual_values[f"pronoun_distribution_{subkey}"].append(
                report["pronoun_distribution"].get(subkey, 0)
            )
            individual_values[f"mentions_pronoun_distribution_{subkey}"].append(
                report["mentions_pronoun_distribution"].get(subkey, 0)
            )
            individual_values[f"feminine_coded_words_pronoun_distribution_{subkey}"].append(
                report["feminine_coded_words_pronoun_distribution"].get(subkey, 0)
            )
            individual_values[f"masculine_coded_words_pronoun_distribution_{subkey}"].append(
                report["masculine_coded_words_pronoun_distribution"].get(subkey, 0)
            )
            individual_values[f"sentiment_by_pronoun_{subkey}"].append(
                report["sentiment_by_pronoun"].get(subkey, 0)
            )

        # Aggregate totals
        accumulated_data["total_actors"] += report["total_actors"]
        accumulated_data["contains_majority_gender_neutral"] += report["contains_majority_gender_neutral"]
        accumulated_data["generic_masculine"] += report["generic_masculine"]
        for key, value in report["pronoun_distribution"].items():
            accumulated_data["pronoun_distribution"][key] += value

        accumulated_data["total_mentions"] += report["total_mentions"]
        for key, value in report["mentions_pronoun_distribution"].items():
            accumulated_data["mentions_pronoun_distribution"][key] += value

        accumulated_data["total_feminine_coded_words"] += report["total_feminine_coded_words"]
        for key, value in report["feminine_coded_words_pronoun_distribution"].items():
            accumulated_data["feminine_coded_words_pronoun_distribution"][key] += value

        accumulated_data["total_masculine_coded_words"] += report["total_masculine_coded_words"]
        for key, value in report["masculine_coded_words_pronoun_distribution"].items():
            accumulated_data["masculine_coded_words_pronoun_distribution"][key] += value

        if isinstance(report["top_pmi_words"], list):
            accumulated_data["pmi_words"].update(dict(report["top_pmi_words"]))
            
        for pronoun in ["she_her", "he_him"]:
            accumulated_data["top_pmi_words_pronoun_distribution"][pronoun].update(
                dict(report["top_pmi_words_pronoun_distribution"].get(pronoun, []))
            )

        for pronoun in ["she_her", "he_him"]:
            if report["sentiment_by_pronoun"][pronoun] is not None:
                accumulated_data["sentiment_by_pronoun"][pronoun].append(report["sentiment_by_pronoun"][pronoun])

        accumulated_data["average_sentiment_all"].append(report["average_sentiment"])

    # Aggregate sentiment
    average_sentiment_all = (
        sum(accumulated_data["average_sentiment_all"]) / len(accumulated_data["average_sentiment_all"])
        if accumulated_data["average_sentiment_all"]
        else None
    )
    sentiment_by_pronoun = {
        pronoun: (sum(values) / len(values)) if values else None
        for pronoun, values in accumulated_data["sentiment_by_pronoun"].items()
    }

    # Convert accumulated data counters to dictionaries
    accumulated_data["pronoun_distribution"] = dict(accumulated_data["pronoun_distribution"])
    accumulated_data["mentions_pronoun_distribution"] = dict(accumulated_data["mentions_pronoun_distribution"])
    accumulated_data["feminine_coded_words_pronoun_distribution"] = dict(
        accumulated_data["feminine_coded_words_pronoun_distribution"]
    )
    accumulated_data["masculine_coded_words_pronoun_distribution"] = dict(
        accumulated_data["masculine_coded_words_pronoun_distribution"]
    )
    accumulated_data["top_pmi_words_pronoun_distribution"] = {
        pronoun: counter.most_common(10)
        for pronoun, counter in accumulated_data["top_pmi_words_pronoun_distribution"].items()
    }

    # Calculate averages and medians
    df = pd.DataFrame(individual_values)
    averages = df.mean(numeric_only=True).to_dict()
    medians = df.median(numeric_only=True).to_dict()

    # Prepare the aggregated report
    aggregated_report = {
        "total_actors": accumulated_data["total_actors"],
        "pronoun_distribution": accumulated_data["pronoun_distribution"],
        "total_mentions": accumulated_data["total_mentions"],
        "mentions_pronoun_distribution": accumulated_data["mentions_pronoun_distribution"],
        "total_feminine_coded_words": accumulated_data["total_feminine_coded_words"],
        "feminine_coded_words_pronoun_distribution": accumulated_data["feminine_coded_words_pronoun_distribution"],
        "total_masculine_coded_words": accumulated_data["total_masculine_coded_words"],
        "masculine_coded_words_pronoun_distribution": accumulated_data["masculine_coded_words_pronoun_distribution"],
        "contains_majority_gender_neutral": docs_with_majority_gender_neutral,
        "generic_masculine": docs_with_generic_masculine,
        "average_metrics": averages,
        "median_metrics": medians,
        "top_pmi_words": accumulated_data["pmi_words"].most_common(10),
        "top_pmi_words_pronoun_distribution": accumulated_data["top_pmi_words_pronoun_distribution"],
        "average_sentiment_all": average_sentiment_all,
        "sentiment_by_pronoun": sentiment_by_pronoun
    }

    return aggregated_report, individual_values


def visualise_boxplot(data, title, xlabel, output_file, exclude_outliers=True, xlim=None):
    """
    Create and save a boxplot for the given data, optionally excluding the top 10 outliers and adjusting the X-axis range.

    Args:
        data (dict): Data to plot. Keys are groups, and values are lists of values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        output_file (str): Path to save the plot.
        exclude_outliers (bool): Whether to exclude the top 10 outliers for each group.
        xlim (tuple): A tuple specifying the X-axis range (min, max). If None, the range is auto-determined.
    """

    def remove_outliers(values):
        """
        Helper function to remove the top 10 outliers from a list of data.
        """
        filtered_values = [v for v in values if v is not None]
        if len(filtered_values) > 10:
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
    plt.figure(figsize=(20, 15))
    df = pd.DataFrame({key: pd.Series(values) for key, values in filtered_data.items()}).melt(var_name="Group",
                                                                                              value_name="Values")

    sns.boxplot(data=df, x="Values", y="Group", orient="h", showfliers=False)

    plt.title(' ', fontsize=40)
    plt.xlabel(xlabel, fontsize=35)
    plt.ylabel('Pronouns', fontsize=35)

    # Set X-axis limits if provided
    if xlim:
        plt.xlim(xlim)

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)

    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"[INFO] Saved boxplot to {output_file}.")


def visualise_barplot(data, title, xlabel, output_file):
    """
    Create and save a barplot for the given binary data.

    Args:
        data (list): List of boolean values (True/False).
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        output_file (str): Path to save the plot.
    """
    # Count occurrences of True and False
    counts = {"True": data.count(True), "False": data.count(False)}

    # Create the barplot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="muted")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"[INFO] Saved barplot to {output_file}.")


def visualise_aggregated_report(individual_values, output_dir):
    """
    Generate visualisations for all individual values in the aggregated report.

    Args:
        individual_values (dict): Dictionary of individual metrics.
        output_dir (str): Directory to save the visualisations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pronoun and sentiment-related metrics with custom X-axis ranges
    pronoun_metrics = {
        "Pronoun Distribution": {
            "total": individual_values["total_actors"],
            "she_her": individual_values["pronoun_distribution_she_her"],
            "he_him": individual_values["pronoun_distribution_he_him"]
        },
        "Mentions by Pronouns": {
            "total": individual_values["total_mentions"],
            "she_her": individual_values["mentions_pronoun_distribution_she_her"],
            "he_him": individual_values["mentions_pronoun_distribution_he_him"]
        },
        "Sentiment by Pronouns": {
            "total": individual_values["average_sentiment_all"],
            "she_her": individual_values["sentiment_by_pronoun_she_her"],
            "he_him": individual_values["sentiment_by_pronoun_he_him"]
        }
    }

    xlim_ranges = {
        "GPronoun Distribution": (0, 20),
        "Mentions by Pronouns": (0, 50),
        "Sentiment by Pronouns": (-0.27, 0.27),
    }

    for metric, data in pronoun_metrics.items():
        output_file = os.path.join(output_dir, f"{metric.replace(' ', '_')}_boxplot.png")
        visualise_boxplot(
            data,
            title=f"{metric} Boxplot",
            xlabel=metric,
            output_file=output_file,
            exclude_outliers=True,
            xlim=xlim_ranges.get(metric),
        )

    # Binary metrics (use barplots)
    binary_metrics = {
        "Contains Majority Gender Neutral": individual_values["contains_majority_gender_neutral"],
        "Generic Masculine": individual_values["generic_masculine"],
    }

    for metric, data in binary_metrics.items():
        output_file = os.path.join(output_dir, f"{metric.replace(' ', '_')}_barplot.png")
        visualise_barplot(
            data,
            title=f"{metric} Barplot",
            xlabel=metric,
            output_file=output_file,
        )

    print(f"[INFO] Boxplots and barplots saved in {output_dir}")


def visualise_knowledgebases(temp_dir, results_dir):
    """
    Visualise metrics from loaded knowledgebases.
    """
    print(f"[INFO] Loading knowledgebases from {temp_dir}...")
    knowledgebases = load_all_knowledgebases(temp_dir)

    if not knowledgebases:
        print(f"[INFO] No valid knowledgebases found in {temp_dir}.")
        return

    # Flatten and combine individual metrics from all knowledgebases
    individual_values = {
        "total_actors": [],
        "total_mentions": [],
        "total_feminine_coded_words": [],
        "total_masculine_coded_words": [],
        "contains_majority_gender_neutral": [],
        "generic_masculine": [],
        "pronoun_distribution_she_her": [],
        "pronoun_distribution_he_him": [],
        "mentions_pronoun_distribution_she_her": [],
        "mentions_pronoun_distribution_he_him": [],
        "feminine_coded_words_pronoun_distribution_she_her": [],
        "feminine_coded_words_pronoun_distribution_he_him": [],
        "masculine_coded_words_pronoun_distribution_she_her": [],
        "masculine_coded_words_pronoun_distribution_he_him": [],
        "average_sentiment_all": [],
        "sentiment_by_pronoun_she_her": [],
        "sentiment_by_gpronoun_he_him": []
    }

    # Combine individual knowledgebase values
    for idx, kb in enumerate(knowledgebases):
        if kb.empty:
            print(f"[DEBUG] Knowledgebase {idx} is empty, skipping.")
            continue
        report = generate_individual_report(kb)
        # Update individual values
        individual_values["total_actors"].append(report["total_actors"])
        individual_values["contains_majority_gender_neutral"].append(report["contains_majority_gender_neutral"])
        individual_values["generic_masculine"].append(report["generic_masculine"])
        individual_values["total_mentions"].append(report["total_mentions"])
        individual_values["total_feminine_coded_words"].append(report["total_feminine_coded_words"])
        individual_values["total_masculine_coded_words"].append(report["total_masculine_coded_words"])
        individual_values["average_sentiment_all"].append(report["average_sentiment"])

        for subkey in ["she_her", "he_him"]:
            individual_values[f"pronoun_distribution_{subkey}"].append(
                report["pronoun_distribution"].get(subkey, 0)
            )
            individual_values[f"mentions_pronoun_distribution_{subkey}"].append(
                report["mentions_pronoun_distribution"].get(subkey, 0)
            )
            individual_values[f"feminine_coded_words_pronoun_distribution_{subkey}"].append(
                report["feminine_coded_words_pronoun_distribution"].get(subkey, 0)
            )
            individual_values[f"masculine_coded_words_pronoun_distribution_{subkey}"].append(
                report["masculine_coded_words_pronoun_distribution"].get(subkey, 0)
            )
            individual_values[f"sentiment_by_pronoun_{subkey}"].append(
                report["sentiment_by_pronoun"].get(subkey)
            )

    # Visualise and save the aggregated data
    print(f"[INFO] Saving visualisations to {results_dir}...")
    visualise_aggregated_report(individual_values, results_dir)
    print(f"[INFO] Visualisation complete!")


            


