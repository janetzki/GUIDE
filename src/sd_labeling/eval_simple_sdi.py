import pandas as pd

test_file_num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# load test sets
dfs = []
for i in test_file_num_list:
    dfs.append(pd.read_csv(f"data/2_sd_labeling/test sets/test_set_{i}.csv", usecols=["direct_question", "answer"]))
df_concat = pd.concat(dfs, ignore_index=True)

# count number of 0 and 1 answers
print(df_concat['answer'].value_counts())

# precision = 0.39 (464/1200)
# recall = 1.0
# f1 = 0.56
