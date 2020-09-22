"""
"""
from collections import Counter
import json

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import pycountry

from make_corpus import get_filenames
from emoji_helpfuns import split_by_emoji


def make_loc_dict():
    da_loc = set()
    for fn in ["danish_cities", "danish_islands", "danish_regions", "others"]:
        with open(f"danish_loc/{fn}.txt") as f:
            da_loc.update(set(f.read().split("\n")))
    loc = {loc: "Denmark" for loc in da_loc}
    loc["Norge"] = "Norway"
    loc["Sverige"] = "Sweden"
    return loc


def extract_countries(text, loc_dict={}):
    """
    """
    if text is None:
        return ""

    text = text.lower()

    country_ls = []
    for country in pycountry.countries:
        if country.name.lower() in text:
            country_ls.append(country.name)

    for loc in loc_dict:
        if loc.lower() in text:
            country_ls.append(loc_dict[loc])
    return country_ls


def extract_specific_country(text, country="Denmark"):
    loc_dict = make_loc_dict()
    countries = extract_countries(text, loc_dict)
    if "Denmark" in countries:
        return True
    return False


def create_emoji_count(df):
    counts = Counter()
    for i in df["emoji"]:
        counts += Counter(split_by_emoji(i))
    return counts


files = get_filenames("data", endswith=".json")

counts = Counter()
for i, file in enumerate(files):
    print(f"File {i}/{len(files)}")
    df = pd.read_json(file)
    da = df.loc[df["lang"] == "da"]
    da = da.loc[da["loc"].apply(extract_specific_country)]
    counts += create_emoji_count(df)
    # da.to_csv("da_emoji_corpus")

for emoji, n in counts.most_common(200):
    print(emoji, "\t", n)


count_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
count_df.columns = ["Emoji", "Count"]
count_df = count_df.sort_values(["Count"], ascending=False)
count_df["Smiley Order"] = range(count_df.shape[0])
count_df.head()
sns.lineplot(count_df, x="Smiley Order", y="Count")

count_df["Smiley Order"]

count_df["Emoji"][-30:]

sns.lineplot(x=count_df["Smiley Order"], y=count_df["Count"])
plt.xscale('log')
plt.yscale('log')

plt.line(range(len(counts.keys())), counts.values())
plt.show()


sum(count_df["Count"][:128])/sum(count_df["Count"])
count_df["Emoji"][128:200].values
with open('emoji_usage_dk.json', 'w') as fp:
    json.dump(counts, fp)