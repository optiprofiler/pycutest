import pandas as pd

if __name__ == "__main__":
    dfs = [pd.read_csv(f'probinfo_pycutest_block{i}.csv') for i in range(20)]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv('probinfo_pycutest.csv', index=False)