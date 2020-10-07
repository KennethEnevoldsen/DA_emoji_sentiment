"""
script contain code for plotting
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json
from collections import Counter
from utils import counter_to_df


with open("emoji_counts.json") as f:
    emoji_counts = json.load(f)

df = counter_to_df(emoji_counts)
df = df.reset_index(drop=True)
df["percent"] = df.Count.apply(lambda x: round(x/sum(df.Count)*100, 2))
sum(df.Count)



def plot_emoji_hist():
    # Load Apple Color Emoji font
    prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')
    # Set up plot
    labels, freqs = list(zip(*Counter(emoji_counts).most_common(32)))

    labels = np.array(labels)
    labels[labels == "flag"] = "🏴"
    plt.figure(figsize=(12, 8))
    p1 = plt.bar(np.arange(len(labels)), freqs, 0.8, color="lightblue")
    plt.ylim(0, plt.ylim()[1]+30)
    # Make labels
    for rect1, label in zip(p1, labels):
        height = rect1.get_height()
        plt.annotate(
            label,
            (rect1.get_x() + rect1.get_width()/2, height+5),
            ha="center",
            va="bottom",
            fontsize=20  # ,fontproperties=prop # adding this will use apple emojis
        )
    plt.show()
