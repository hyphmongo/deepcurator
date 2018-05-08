import csv
import math
import pandas as pd


def bayesian_rating(rating, votes, mean, min=30):
    return ((float(votes) / (votes + min)) * rating) + \
        ((float(min) / (votes + min)) * mean)


def min_max(col):
    return (col - col.min())/(col.max() - col.min())


def combined_score(df):
    score = (
        min_max(df['price']) +
        min_max(df['ratio']) +
        min_max(df['weighted_rating'])
    )

    return score / 3


df = pd.read_csv('dataset.csv')

# Create want/have ratio and filter outliers
df['ratio'] = df['want'] / df['have']
q = df["ratio"].quantile(0.99)
df = df[df["ratio"] < q]

# Remove anything with no price, and filter outliers
df = df[df.price != 0]
q = df["price"].quantile(0.99)
df = df[df['price'] < q]

# Add a weighted bayesian for the rating column
mean_rating = df['rating_average'].mean()
df['weighted_rating'] = df.apply(lambda row: bayesian_rating(
    row.rating_average, row.rating_count, mean_rating), axis=1)

# Create normalized (0-1) score for the combined rating, price and ratio columns
df['score'] = min_max(combined_score(df))
df.sort_values('score', ascending=False)

df = df.reset_index(drop=True)
row_count = df.shape[0]

SPLIT_SIZE = 15000
BOTTOM_SPLIT = row_count - SPLIT_SIZE

if row_count < SPLIT_SIZE * 2:
    print ('Not enough data to split')
    quit()

with open('labels.csv', 'w') as file:
    for index, row in df.iterrows():
        if index < SPLIT_SIZE:
            file.write(row['youtube_id'] + ',1\n')
        elif index >= BOTTOM_SPLIT:
            file.write(row['youtube_id'] + ',0\n')
