import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATA_PATH = 'data/raw/complaints.csv'
OUTPUT_PATH = 'data/filtered_complaints.csv'
TARGET_PRODUCTS = [
    "Credit card", 
    "Personal loan", 
    "Savings account", 
    "Money transfers"
]
CHUNK_SIZE = 100000

def run_eda_and_preprocessing():
    print("Processing dataset in chunks...")
    
    product_counts = {}
    narrative_lengths = []
    total_complaints = 0
    missing_narratives = 0
    
    # Initialize output file
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    
    first_chunk = True
    
    try:
        # Use chunks to handle large file
        # usecols=['Product', 'Consumer complaint narrative', 'Complaint ID'] to save memory/speed
        with pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, usecols=['Product', 'Consumer complaint narrative', 'Complaint ID']) as reader:
            for i, chunk in enumerate(reader):
                print(f"Processing chunk {i+1}...")
                
                # EDA Aggregations
                # Product counts
                counts = chunk['Product'].value_counts().to_dict()
                for prod, count in counts.items():
                    product_counts[prod] = product_counts.get(prod, 0) + count
                
                total_complaints += len(chunk)
                missing_narratives += chunk['Consumer complaint narrative'].isna().sum()
                
                # Narrative Length (sample or all? doing all might be slow but feasible in chunks)
                # Let's just do a sample for the histogram to save memory/time if needed, or all. 
                # We can extend a list, but list of millions of ints is fine (MBs).
                # Only for non-null
                lengths = chunk['Consumer complaint narrative'].dropna().astype(str).apply(lambda x: len(x.split())).tolist()
                narrative_lengths.extend(lengths)

                # Preprocessing & Filtering
                # Filter by Product
                chunk_filtered = chunk[chunk['Product'].isin(TARGET_PRODUCTS)].copy()
                
                # Remove empty narratives
                chunk_filtered = chunk_filtered.dropna(subset=['Consumer complaint narrative'])
                
                # Clean text
                if not chunk_filtered.empty:
                     chunk_filtered['Consumer complaint narrative'] = chunk_filtered['Consumer complaint narrative'].apply(
                        lambda x: x.lower().replace("i am writing to file a complaint", "").strip() if isinstance(x, str) else ""
                     )
                     
                     # Append to output
                     mode = 'w' if first_chunk else 'a'
                     header = first_chunk
                     chunk_filtered.to_csv(OUTPUT_PATH, mode=mode, header=header, index=False)
                     first_chunk = False

    except FileNotFoundError:
        print(f"Error: File not found at {DATA_PATH}")
        return
    except Exception as e:
        print(f"Error processing chunks: {e}")
        # Consider re-raising or logging
        raise

    print("Data processing complete.")

    # EDA Visualizations
    print("Generating EDA plots...")
    
    # Product Distribution
    plt.figure(figsize=(10, 6))
    series_counts = pd.Series(product_counts).sort_values(ascending=False)
    # Filter top N or just show all if not too many
    if len(series_counts) > 20:
        series_counts = series_counts.head(20)
    
    sns.barplot(
        y=series_counts.index, 
        x=series_counts.values
    )
    plt.title('Distribution of Complaints by Product (Top 20)')
    plt.xlabel('Number of Complaints')
    plt.tight_layout()
    plt.savefig('notebooks/product_distribution.png')
    print("Saved product_distribution.png")

    # Narrative Length
    # If too many points, sample for plot
    if len(narrative_lengths) > 100000:
        import random
        plot_lengths = random.sample(narrative_lengths, 100000)
    else:
        plot_lengths = narrative_lengths
        
    plt.figure(figsize=(10, 6))
    sns.histplot(plot_lengths, bins=50, kde=True)
    plt.title('Distribution of Complaint Narrative Length (Word Count)')
    plt.xlabel('Word Count')
    plt.tight_layout()
    plt.savefig('notebooks/narrative_length_distribution.png')
    print("Saved narrative_length_distribution.png")

    print(f"Total complaints: {total_complaints}")
    print(f"Complaints with missing narratives: {missing_narratives} ({missing_narratives/total_complaints:.2%})")
    print("Done.")

if __name__ == "__main__":
    if not os.path.exists('data/raw/complaints.csv'):
        if os.path.exists('../data/raw/complaints.csv'):
            os.chdir('..')
    
    run_eda_and_preprocessing()
