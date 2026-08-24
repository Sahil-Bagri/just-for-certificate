import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv('scraped_data.csv')

# 2. Inspect structure and data types
print("--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# 3. Clean duplicates and missing records
df.drop_duplicates(subset=['text'], inplace=True)
df['tags'] = df['tags'].fillna('Unknown')

# 4. Feature engineering for text metrics
df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

print("\n--- Summary Statistics ---")
print(df[['char_count', 'word_count']].describe())

# 5. Anomaly / Outlier Detection using Interquartile Range (IQR)
Q1 = df['word_count'].quantile(0.25)
Q3 = df['word_count'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['word_count'] < lower_bound) | (df['word_count'] > upper_bound)]
print(f"\nDetected {len(outliers)} word-count outliers based on 1.5*IQR threshold.")

# Save cleaned dataset for downstream analysis
df.to_csv('cleaned_data.csv', index=False)
