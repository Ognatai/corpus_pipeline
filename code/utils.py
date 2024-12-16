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
    tmp_path = temp_dir / str(year)
    try:
        # Count total number of texts by counting the knowledge base files
        total_texts = len(list(tmp_path.glob("knowledgebase_*.pkl")))

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(f"Aggregated Report for {year}\n")
            file.write("=" * 60 + "\n\n")

            file.write(f"Total Texts: {total_texts}\n")
            file.write(f"Total Actors: {aggregated_report['total_actors']}\n")
            file.write(f"Gender Distribution: {aggregated_report['gender_distribution']}\n")
            file.write(f"Total Mentions: {aggregated_report['total_mentions']}\n")
            file.write(f"Mentions by Gender: {aggregated_report['mentions_gender_distribution']}\n\n")
            
            file.write("Minimal Metrics:\n")
            for metric, value in aggregated_report['min_metrics'].items():
                file.write(f"  {metric}: {value:.2f}\n")
            
            file.write("\nMaximal Metrics:\n")
            for metric, value in aggregated_report['max_metrics'].items():
                file.write(f"  {metric}: {value:.2f}\n")
            
            file.write("\nMean Metrics:\n")
            for metric, value in aggregated_report['average_metrics'].items():
                file.write(f"  {metric}: {value:.2f}\n")

            file.write("\nMedian Metrics:\n")
            for metric, value in aggregated_report['median_metrics'].items():
                file.write(f"  {metric}: {value:.2f}\n")
                
            file.write("\nTop PMI Words Table:\n")
            file.write(f"{'All':<25} {'Woman':<25} {'Man':<25} {'Unknown':<25}\n")
            file.write("-" * 100 + "\n")

            # Prepare lists of words and scores for each category
            top_all = aggregated_report["top_pmi_words"]
            top_woman = aggregated_report["top_pmi_words_gender_distribution"].get("woman", [])
            top_man = aggregated_report["top_pmi_words_gender_distribution"].get("man", [])
            top_unknown = aggregated_report["top_pmi_words_gender_distribution"].get("unknown", [])

            # Ensure all lists have the same length
            max_length = max(len(top_all), len(top_woman), len(top_man), len(top_unknown))
            top_all += [("", 0)] * (max_length - len(top_all))
            top_woman += [("", 0)] * (max_length - len(top_woman))
            top_man += [("", 0)] * (max_length - len(top_man))
            top_unknown += [("", 0)] * (max_length - len(top_unknown))

            # Write rows for the table
            for i in range(max_length):
                all_word, all_score = top_all[i]
                woman_word, woman_score = top_woman[i]
                man_word, man_score = top_man[i]
                unknown_word, unknown_score = top_unknown[i]

                file.write(
                    f"{all_word} ({all_score:<.2f})".ljust(25)
                    + f"{woman_word} ({woman_score:<.2f})".ljust(25)
                    + f"{man_word} ({man_score:<.2f})".ljust(25)
                    + f"{unknown_word} ({unknown_score:<.2f})".ljust(25)
                    + "\n"
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
        gender_distribution = knowledgebase["gender"].value_counts().to_dict()
        total_mentions = knowledgebase["mention_count"].sum()
        mentions_gender_distribution = knowledgebase.groupby("gender")["mention_count"].sum().to_dict()
        total_feminine_coded_words = knowledgebase["feminine_coded_words"].sum()
        feminine_coded_words_gender_distribution = knowledgebase.groupby("gender")[
            "feminine_coded_words"].sum().to_dict()
        total_masculine_coded_words = knowledgebase["masculine_coded_words"].sum()
        masculine_coded_words_gender_distribution = knowledgebase.groupby("gender")[
            "masculine_coded_words"].sum().to_dict()
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
        top_pmi_words_gender_distribution = {}
        for gender in ["woman", "man", "unknown"]:
            gender_data = knowledgebase[knowledgebase["gender"] == gender]
            pmi_counter = Counter(
                word
                for pmi in gender_data["pmi"]
                for word in (pmi or {})
            )
            top_pmi_words_gender_distribution[gender] = pmi_counter.most_common(10)

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
            "top_pmi_words_gender_distribution": top_pmi_words_gender_distribution,
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
        "top_pmi_words_gender_distribution":  {
            "woman": Counter(),
            "man": Counter(),
            "unknown": Counter()
        },
        "average_sentiment_all": [],
        "sentiment_by_gender": {
            "woman": [],
            "man": [],
            "unknown": []
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
            accumulated_data["top_pmi_words_gender_distribution"][gender].update(
                dict(report["top_pmi_words_gender_distribution"].get(gender, []))
            )

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
    accumulated_data["top_pmi_words_gender_distribution"] = {
        gender: counter.most_common(10)
        for gender, counter in accumulated_data["top_pmi_words_gender_distribution"].items()
    }

    # Calculate mins, max, averages and medians
    df = pd.DataFrame(individual_values)
    averages = df.mean(numeric_only=True).to_dict()
    medians = df.median(numeric_only=True).to_dict()
    mins = df.min(numeric_only=True).to_dict()
    maxs = df.max(numeric_only=True).to_dict()

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
        "min_metrics": mins,
        "max_metrics": maxs,
        "top_pmi_words": accumulated_data["pmi_words"].most_common(10),
        "top_pmi_words_gender_distribution": accumulated_data["top_pmi_words_gender_distribution"],
        "average_sentiment_all": average_sentiment_all,
        "sentiment_by_gender": sentiment_by_gender
    }

    return aggregated_report, individual_values


def visualise_boxplot(data, title, xlabel, output_file, exclude_outliers=False, xlim=None):
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

    sns.boxplot(data=df, x="Values", y="Group", orient="h")

    plt.title(' ', fontsize=30)
    plt.xlabel(xlabel, fontsize=55)
    plt.ylabel('Gender', fontsize=55)

    # Set X-axis limits if provided
    if xlim:
        plt.xlim(xlim)

    plt.xticks(fontsize=50, rotation=90)
    plt.yticks(fontsize=50)

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

    # Gender and sentiment-related metrics with custom X-axis ranges
    gender_metrics = {
        "Gender Distribution": {
            "total": individual_values["total_actors"],
            "woman": individual_values["gender_distribution_woman"],
            "man": individual_values["gender_distribution_man"],
            "unknown": individual_values["gender_distribution_unknown"],
        },
        "Mentions by Gender": {
            "total": individual_values["total_mentions"],
            "woman": individual_values["mentions_gender_distribution_woman"],
            "man": individual_values["mentions_gender_distribution_man"],
            "unknown": individual_values["mentions_gender_distribution_unknown"],
        },
        "Sentiment by Gender": {
            "total": individual_values["average_sentiment_all"],
            "woman": individual_values["sentiment_by_gender_woman"],
            "man": individual_values["sentiment_by_gender_man"],
            "unknown": individual_values["sentiment_by_gender_unknown"],
        }
    }

    xlim_ranges = {
        "Gender Distribution": (0, 20),
        "Mentions by Gender": (0, 50),
        "Sentiment by Gender": (-0.27, 0.27),
    }

    for metric, data in gender_metrics.items():
        output_file = os.path.join(output_dir, f"{metric.replace(' ', '_')}_boxplot.png")
        visualise_boxplot(
            data,
            title=f"{metric} Boxplot",
            xlabel=metric,
            output_file=output_file,
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
                report["sentiment_by_gender"].get(subkey)
            )

    # Visualise and save the aggregated data
    print(f"[INFO] Saving visualisations to {results_dir}...")
    visualise_aggregated_report(individual_values, results_dir)
    print(f"[INFO] Visualisation complete!")


            


