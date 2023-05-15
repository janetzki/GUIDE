import pandas as pd

test_file_num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# load test sets
dfs = []
for i in test_file_num_list:
    dfs.append(pd.read_csv(f"data/2_sd_labeling/test sets/test_set_{i}.csv", usecols=["direct_question", "answer"]))
df_concat = pd.concat(dfs, ignore_index=True)

# write answer column from df_concat into answer column of data/2_sd_labeling/matched_questions.xlsx
df_main = pd.read_excel("data/2_sd_labeling/matched_questions.xlsx", nrows=156501)

# build an index of the direct_question column
df_main.set_index('direct_question', inplace=True)

# JOIN LEFT: write every answer value from df_concat into df, if the direct_question is the same
for index, row in df_concat.iterrows():
    df_main.at[row['direct_question'], 'answer'] = row['answer']

# add direct_question column
df_main.reset_index(inplace=True)

df_main.to_excel("data/2_sd_labeling/matched_questions_merged.xlsx", index=False)
print("Done.")