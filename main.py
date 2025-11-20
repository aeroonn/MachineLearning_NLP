# 2. Load Data (BBC News)
from datasets import load_dataset

dataset = load_dataset("SetFit/bbc-news")
docs = dataset["train"]["text"]
# BBC News has 5 categories: business, entertainment, politics, sport, tech
categories = dataset["train"]["label_text"]

# 3. REPRODUCTION: Standard BERTopic (As per paper)
from bertopic import BERTopic

# We set a seed for reproducibility
topic_model_baseline = BERTopic(language="english", calculate_probabilities=True, verbose=True)
topics_base, probs_base = topic_model_baseline.fit_transform(docs)

print("--- BASELINE RESULTS (c-TF-IDF) ---")
print(topic_model_baseline.get_topic_info().head(5))

# 4. IMPROVEMENT: Better Representation (KeyBERTInspired)
# This addresses the weakness that c-TF-IDF favors frequency over meaning.
from bertopic.representation import KeyBERTInspired

# We keep the same clusters, just update the representation!
representation_model = KeyBERTInspired()

# We update the existing model (this saves time and proves the modularity concept)
topic_model_baseline.update_topics(docs, representation_model=representation_model)

print("\n--- IMPROVED RESULTS (KeyBERT) ---")
print(topic_model_baseline.get_topic_info().head(5))

# 5. VISUALIZATION (The "Impact" for your poster)
# Visualize the clusters - take a screenshot of this for your presentation
topic_model_baseline.visualize_topics()

# Get the info from the model
info_df = topic_model_baseline.get_topic_info()

# Select only the columns we care about: The Topic ID and the Keywords
# 'Representation' holds the list of keywords
clean_results = info_df[["Topic", "Count", "Representation"]]

print("--- THE HIDDEN RESULTS ---")
print(clean_results.head(10))

import pandas as pd

# 1. Get the IMPROVED representation (KeyBERT)
# The fix: we iterate through the tuples (word, score) and pick just the word [0]
new_topics = topic_model_baseline.get_topics()
new_keywords = []
for topic_id in new_topics:
    # Get the list of (word, score) tuples for this topic
    topic_list = new_topics[topic_id]
    # Extract just the word (the first element of the tuple)
    words = [pair[0] for pair in topic_list[:5]]
    new_keywords.append(", ".join(words))

# 2. QUICKLY RE-CALCULATE THE BASELINE (c-TF-IDF)
# We revert the representation to get the old style back
topic_model_baseline.update_topics(docs, representation_model=None)

# Get the OLD keywords (Standard BERTopic usually returns tuples too)
old_topics = topic_model_baseline.get_topics()
old_keywords = []
for topic_id in old_topics:
    topic_list = old_topics[topic_id]
    words = [pair[0] for pair in topic_list[:5]]
    old_keywords.append(", ".join(words))

# 3. Create the COMPARISON TABLE
# We use a dictionary to ensure we match Topic ID to Topic ID correctly
df_compare = pd.DataFrame({
    "Topic": list(new_topics.keys()),
    "Baseline_Keywords (Old)": old_keywords,
    "Improved_Keywords (New)": new_keywords
})

# Filter out Topic -1 (Noise) and sort by Topic ID
df_compare = df_compare[df_compare["Topic"] != -1].sort_values("Topic")

# Show the table
pd.set_option('display.max_colwidth', None)
print(df_compare.head(10))