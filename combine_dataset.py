import pandas as pd

#File paths for train test and validation as dataset was seperate
file1 = "train.csv"
file2 = "test.csv"
file3 = "validation.csv"

#Reading CSV files into dataframes
train_df = pd.read_csv(file1)
test_df = pd.read_csv(file2)
validation_df = pd.read_csv(file3)

#Combining into one DataFrame
combined_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)

#Randomly select 10,000 rows as dataset was too large
sampled_df = combined_df.sample(n=10000, random_state=42)

#Saving to new CSV
sampled_df.to_csv("clipped_dataset.csv", index=False)

print("Combined and saved 10,000 random rows to 'combined_sampled.csv'.")
