import pandas as pd

# 加载数据集
path1 = "/home/pptan/Codex/dual-channel-for-sarcasm-main/data/IAC-V1_1/train_augmented_synonyms.csv"
path2 = "/home/pptan/Codex/dual-channel-for-sarcasm-main/data/IAC-V1_1/train_1.csv"

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

# 由于原始文本已在两个文件中重复，我们将合并这两个数据集，然后去除完全重复的行
combined_df = pd.concat([df1, df2]).drop_duplicates(subset=['text'], keep='first')

# 保存合并后的数据集
new_path = "/home/pptan/Codex/dual-channel-for-sarcasm-main/data/IAC-V1_1/train_1.csv"
combined_df.to_csv(new_path, index=False)

print(f"Saved combined data to {new_path}")
print(f"The combined dataset now contains {combined_df.shape[0]} entries.")
