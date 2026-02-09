import pandas as pd
import os

def merge_txt_files(pattern, output_file, num_blocks=20):
    """Merge multiple txt files containing space-separated problem names."""
    all_problems = set()
    for i in range(num_blocks):
        filename = pattern.format(i)
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    content = f.read().strip()
                    if content:
                        problems = content.split()
                        all_problems.update(problems)
                        print(f"  Found {len(problems)} problems in {filename}")
            except Exception as e:
                print(f"  Error reading {filename}: {e}")
    
    if all_problems:
        with open(output_file, 'w') as f:
            f.write(' '.join(sorted(all_problems)))
        print(f"  Total: {len(all_problems)} unique problems saved to {output_file}")
    else:
        print(f"  No problems found for {output_file}")
    
    return all_problems


if __name__ == "__main__":
    num_blocks = 20
    
    # Merge CSV files
    print("Merging CSV files...")
    dfs = []
    for i in range(num_blocks):
        csv_file = f'probinfo_pycutest_block{i}.csv'
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
                print(f"  Loaded {len(df)} problems from {csv_file}")
            except Exception as e:
                print(f"  Error reading {csv_file}: {e}")
    
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        # Remove duplicates based on problem_name
        merged_df = merged_df.drop_duplicates(subset=['problem_name'])
        merged_df.to_csv('probinfo_pycutest.csv', index=False)
        print(f"  Total: {len(merged_df)} unique problems saved to probinfo_pycutest.csv")
    else:
        print("  No CSV files found!")
    
    # Merge failed problems files
    print("\nMerging failed problems files...")
    failed_problems = merge_txt_files(
        'failed_problems_pycutest_block{}.txt',
        'failed_problems_pycutest.txt',
        num_blocks
    )
    
    # Merge feasibility files
    print("\nMerging feasibility files...")
    merge_txt_files(
        'feasibility_pycutest_block{}.txt',
        'feasibility_pycutest.txt',
        num_blocks
    )
    
    # Merge timeout files
    print("\nMerging timeout files...")
    merge_txt_files(
        'timeout_problems_pycutest_block{}.txt',
        'timeout_problems_pycutest.txt',
        num_blocks
    )
    
    print("\nMerge complete!")