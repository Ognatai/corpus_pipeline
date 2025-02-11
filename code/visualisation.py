import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Path to the main folder containing subfolders
data_path = "results"
output_path = "figures"
pmi_output_file = "top_pmi_words.txt"
os.makedirs(output_path, exist_ok=True)

# Set font size
plt.rc("font", size=15)

# Prepare data collection
mention_counts = []
actor_counts = []
sentiment_avg = []
pmi_words = []

# Iterate through each subfolder and load the .pkl file
for subfolder in sorted(os.listdir(data_path)):
    folder_path = os.path.join(data_path, subfolder)
    if os.path.isdir(folder_path) and subfolder.isdigit() and int(subfolder) >= 1986:
        for file in os.listdir(folder_path):
            if file.endswith(".pkl"):
                file_path = os.path.join(folder_path, file)
                df = pd.read_pickle(file_path)
                df["category"] = df["main_pronoun"].map({"she_her": "woman", "he_him": "man"})
                df["year"] = int(subfolder)
                
                # Ensure mention_count is numeric
                df["mention_count"] = pd.to_numeric(df["mention_count"], errors="coerce")
                
                # Extract sentiment values
                sentiment_avg.append(
                    df.groupby("category", as_index=False)["sentiment"]
                    .apply(lambda x: x.dropna().apply(lambda y: y["average_sentiment"] if isinstance(y, dict) and "average_sentiment" in y else None).mean())
                    .assign(year=int(subfolder))
                )
                
                # Compute percentage-based mention counts
                mention_count_df = df.groupby("category", as_index=False)["mention_count"].sum()
                mention_count_df["year"] = int(subfolder)
                mention_count_df["mention_count"] = (mention_count_df["mention_count"] / mention_count_df["mention_count"].sum()) * 100
                mention_counts.append(mention_count_df)
                
                # Compute percentage-based actor counts
                actor_count_df = df.groupby("category", as_index=False).size().rename(columns={"size": "count"})
                actor_count_df["year"] = int(subfolder)
                actor_count_df["count"] = (actor_count_df["count"] / actor_count_df["count"].sum()) * 100
                actor_counts.append(actor_count_df)
                
                # Compute top PMI words
                top_pmi_words_pronoun_distribution = {}
                for pronoun in ["she_her", "he_him"]:
                    pronoun_data = df[df["main_pronoun"] == pronoun]
                    pmi_counter = Counter(
                        word
                        for pmi in pronoun_data["pmi"]
                        for word in (pmi or {})
                    )
                    top_pmi_words_pronoun_distribution[pronoun] = pmi_counter.most_common(10)
                
                pmi_words.append({"year": int(subfolder), "pmi_words": top_pmi_words_pronoun_distribution})

# Concatenate results
mention_counts = pd.concat(mention_counts, ignore_index=True)
actor_counts = pd.concat(actor_counts, ignore_index=True)
sentiment_avg = pd.concat(sentiment_avg, ignore_index=True)

# Save PMI words to a text file
with open(pmi_output_file, "w", encoding="utf-8") as f:
    for entry in pmi_words:
        f.write(f"Year: {entry['year']}\n")
        for pronoun, words in entry["pmi_words"].items():
            category = "woman" if pronoun == "she_her" else "man"
            f.write(f"  {category}: {', '.join([word for word, _ in words])}\n")
        f.write("\n")

# Adjust y-axis for sentiment
sentiment_range = (sentiment_avg["sentiment"].min(), sentiment_avg["sentiment"].max())

# Plot mention percentages
plt.figure(figsize=(10, 5))
sns.lineplot(data=mention_counts, x="year", y="mention_count", hue="category", hue_order=["woman", "man"])
plt.xlabel("Year")
plt.xticks(mention_counts['year'][mention_counts['year'] % 5 == 0], rotation=45, ha='right')
plt.ylabel("Percentage of Mentions")
plt.legend(title="Category")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "mention_counts.png"))
plt.close()

# Plot actor percentages
plt.figure(figsize=(10, 5))
sns.lineplot(data=actor_counts, x="year", y="count", hue="category", hue_order=["woman", "man"])
plt.xlabel("Year")
plt.xticks(actor_counts['year'][actor_counts['year'] % 5 == 0], rotation=45, ha='right')
plt.ylabel("Percentage of Actors")
plt.legend(title="Category")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "actor_counts.png"))
plt.close()

# Plot combined actor and mention percentages
plt.figure(figsize=(10, 5))
sns.lineplot(data=mention_counts, x="year", y="mention_count", hue="category", hue_order=["woman", "man"], linestyle="--")
sns.lineplot(data=actor_counts, x="year", y="count", hue="category", hue_order=["woman", "man"], linestyle="-")
plt.xlabel("Year")
plt.xticks(mention_counts['year'][mention_counts['year'] % 5 == 0], rotation=45, ha='right')
plt.ylabel("Percentage")
plt.legend(title="Legend", loc='center left', bbox_to_anchor=(1, 0.5), handles=[
    plt.Line2D([0], [0], color='blue', linestyle='--', label='Women Mentions'),
    plt.Line2D([0], [0], color='orange', linestyle='--', label='Men Mentions'),
    plt.Line2D([0], [0], color='blue', linestyle='-', label='Women Actors'),
    plt.Line2D([0], [0], color='orange', linestyle='-', label='Men Actors')
])
plt.tight_layout()
plt.savefig(os.path.join(output_path, "combined_mentions_actors.png"))
plt.close()

# Plot sentiment
plt.figure(figsize=(10, 5))
sns.lineplot(data=sentiment_avg, x="year", y="sentiment", hue="category", hue_order=["woman", "man"])
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Year")
plt.xticks(sentiment_avg['year'][sentiment_avg['year'] % 5 == 0], rotation=45, ha='right')
plt.ylabel("Average Sentiment")
plt.ylim(sentiment_range[0] - 0.05, sentiment_range[1] + 0.05)
plt.legend(title="Category")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "sentiment_avg.png"))
plt.close()
