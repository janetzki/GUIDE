import pandas as pd

test_file_num_start = 10
test_file_num_end = test_file_num_start + 3

# # select 3 random rows from df and save them to a csv file
# df = pd.read_excel("data/2_sd_labeling/matched_questions.xlsx", nrows=156501)
#
# for i in range(test_file_num_start, test_file_num_end):
#     df_sample = df.sample(n=100, random_state=i)
#     df_sample.to_csv(f"data/2_sd_labeling/test sets/test_set_{i}.csv", index=False)

# identify the number of duplicates
dfs = []
for i in range(1, test_file_num_end):
    dfs.append(pd.read_csv(f"data/2_sd_labeling/test sets/test_set_{i}.csv"))
df_concat = pd.concat(dfs, ignore_index=True)
print(df_concat['direct_question'].duplicated().sum())

# print duplicates
print(df_concat[df_concat['direct_question'].duplicated()])