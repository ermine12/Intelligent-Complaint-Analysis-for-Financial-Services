import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import re

# Configuration
DATA_PATH = 'data/raw/complaints.csv'
OUTPUT_DIR = 'notebooks'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load the complaints data."""
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return None
    try:
        # Load in chunks if large, but for EDA, try full load first
        df = pd.read_csv(DATA_PATH, low_memory=False)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_statistics(df):
    """Print basic statistics."""
    print("\n=== Basic Statistics ===")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())

    # Categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical columns: {list(cat_cols)}")

    # Numerical if any
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        print(f"Numerical columns: {list(num_cols)}")
        print(df[num_cols].describe())

def visualize_distributions(df):
    """Visualize distributions of key columns."""
    print("\n=== Visualizing Distributions ===")

    # Product distribution
    if 'Product' in df.columns:
        plt.figure(figsize=(10, 6))
        top_products = df['Product'].value_counts().head(10)
        sns.barplot(y=top_products.index, x=top_products.values)
        plt.title('Top 10 Products by Number of Complaints')
        plt.xlabel('Number of Complaints')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/product_distribution.png')
        plt.show()
        print("Saved product_distribution.png")

    # Issue distribution
    if 'Issue' in df.columns:
        plt.figure(figsize=(10, 6))
        top_issues = df['Issue'].value_counts().head(10)
        sns.barplot(y=top_issues.index, x=top_issues.values)
        plt.title('Top 10 Issues by Number of Complaints')
        plt.xlabel('Number of Complaints')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/issue_distribution.png')
        plt.show()
        print("Saved issue_distribution.png")

    # Company distribution
    if 'Company' in df.columns:
        plt.figure(figsize=(10, 6))
        top_companies = df['Company'].value_counts().head(10)
        sns.barplot(y=top_companies.index, x=top_companies.values)
        plt.title('Top 10 Companies by Number of Complaints')
        plt.xlabel('Number of Complaints')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/company_distribution.png')
        plt.show()
        print("Saved company_distribution.png")

    # State distribution
    if 'State' in df.columns:
        plt.figure(figsize=(10, 6))
        top_states = df['State'].value_counts().head(10)
        sns.barplot(y=top_states.index, x=top_states.values)
        plt.title('Top 10 States by Number of Complaints')
        plt.xlabel('Number of Complaints')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/state_distribution.png')
        plt.show()
        print("Saved state_distribution.png")

def text_analysis(df):
    """Analyze text data in narratives."""
    print("\n=== Text Analysis ===")
    if 'Consumer complaint narrative' not in df.columns:
        print("No 'Consumer complaint narrative' column found.")
        return

    narratives = df['Consumer complaint narrative'].dropna().astype(str)

    # Narrative lengths
    lengths = narratives.apply(lambda x: len(x.split()))
    print(f"Narrative lengths: Mean={lengths.mean():.2f}, Median={lengths.median():.2f}, Max={lengths.max()}")

    plt.figure(figsize=(10, 6))
    # Sample if too many
    if len(lengths) > 50000:
        lengths = lengths.sample(50000)
    sns.histplot(lengths, bins=50, kde=True)
    plt.title('Distribution of Complaint Narrative Length (Word Count)')
    plt.xlabel('Word Count')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/narrative_length_distribution.png')
    plt.show()
    print("Saved narrative_length_distribution.png")

    # Common words
    all_words = []
    for text in narratives.sample(min(10000, len(narratives))):  # Sample for speed
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)

    common_words = Counter(all_words).most_common(20)
    print("Top 20 common words:")
    for word, count in common_words:
        print(f"{word}: {count}")

    # Word frequency bar plot
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words))
    plt.title('Top 20 Most Common Words in Narratives')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/common_words.png')
    plt.show()
    print("Saved common_words.png")

def correlation_analysis(df):
    """Analyze correlations if numerical columns exist."""
    print("\n=== Correlation Analysis ===")
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png')
        plt.show()
        print("Saved correlation_matrix.png")
    else:
        print("Not enough numerical columns for correlation analysis.")

def run_eda():
    """Run the full EDA."""
    df = load_data()
    if df is None:
        return

    basic_statistics(df)
    visualize_distributions(df)
    text_analysis(df)
    correlation_analysis(df)

    print("\nEDA complete. Plots saved to notebooks/")

if __name__ == "__main__":
    run_eda()