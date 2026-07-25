


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# 1. Load the Retail Dataset
file_path = "retail_large_dataset.csv"
df = pd.read_csv(file_path)

print("--- Dataset Initial Overview ---")
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
print(df.info())

# Create an output directory for saving visuals
output_dir = "retail_eda_plots"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# OBJECTIVE 1 & 4: UNIVARIATE ANALYSIS, SKEWNESS & KURTOSIS
# =====================================================================

print("\n=== Processing Objective 1 & 4: Univariate Analysis ===")

# A. Numerical Columns Univariate Analysis
num_cols = [
    "product_price",
    "final_price",
    "discount_percentage",
    "quantity",
    "age",
    "delivery_days",
]

# Ensure they exist in the uploaded dataframe before analyzing
num_cols = [col for col in num_cols if col in df.columns]

for col in num_cols:
    # Calculating key summary statistics
    mean_val = df[col].mean()
    median_val = df[col].median()
    std_val = df[col].std()
    min_val = df[col].min()
    max_val = df[col].max()

    # Measuring Skewness and Kurtosis
    col_skew = stats.skew(df[col].dropna())
    col_kurt = stats.kurtosis(
        df[col].dropna()
    )  # Excess Kurtosis (Fisher's definition)

    print(f"\nSummary Statistics for Column: {col}")
    print(
        f"  Mean: {mean_val:.2f} | Median: {median_val:.2f} | Std Dev: {std_val:.2f}"
    )
    print(f"  Min: {min_val:.2f} | Max: {max_val:.2f}")
    print(f"  Skewness: {col_skew:.2f} | Kurtosis: {col_kurt:.2f}")

    # Plot Distribution (Histogram + KDE)

    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, color="teal")
    plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
    plt.axvline(
        median_val, color="green", linestyle="-.", label=f"Median: {median_val:.2f}"
    )
    plt.title(
        f"Distribution of {col}\nSkewness: {col_skew:.2f} | Kurtosis: {col_kurt:.2f}"
    )
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/univariate_num_{col}.png")
    plt.close()

    print("="*60)


# B. Categorical Columns Univariate Analysis (Frequency Distributions)
cat_cols = [
    "product_category",
    "customer_segment",
    "payment_method",
    "return_status",
]
cat_cols = [col for col in cat_cols if col in df.columns]

for col in cat_cols:
    print(f"\nFrequency Distribution for Column: {col}")
    freq = df[col].value_counts()
    percentage = df[col].value_counts(normalize=True) * 100
    freq_df = pd.DataFrame({"Count": freq, "Percentage (%)": percentage})
    print(freq_df)

    # Plot Frequency Distribution Bar Chart
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=df,
        x=col,
        order=freq.index,
        hue=col,
        palette="viridis",
        legend=False,
    )
    plt.title(f"Frequency Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/univariate_cat_{col}.png")
    plt.close()

    print("*"*60)

# =====================================================================
# OBJECTIVE 2: DETECT AND INTERPRET OUTLIERS
# =====================================================================
print("\n=== Processing Objective 2: Outlier Detection ===")

for col in ["product_price", "final_price"]:
    if col in df.columns:
        # Interquartile Range (IQR) Method
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        print(f"\nOutlier Evaluation for {col}:")
        print(f"  IQR: {iqr:.2f} | Bounds: [{lower_bound:.2f} to {upper_bound:.2f}]")
        print(
            f"  Detected Outliers: {len(outliers)} rows out of {len(df)} ({len(outliers)/len(df)*100:.2f}%)"
        )

        # Plot Boxplot for Outliers Visual
        plt.figure(figsize=(8, 3))
        sns.boxplot(x=df[col], color="coral")
        plt.title(f"Boxplot for Outlier Detection in {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/outliers_{col}.png")
        plt.close()

        print("@"*60)

# =====================================================================
# OBJECTIVE 3: MEASURE CORRELATION BETWEEN NUMERICAL VARIABLES
# =====================================================================
print("\n=== Processing Objective 3: Correlation Measurement ===")

if len(num_cols) > 1:
    # Compute Pearson correlation matrix
    corr_matrix = df[num_cols].corr(method="pearson")
    print("\nPearson Correlation Matrix:")
    print(corr_matrix.round(2))

    # Generate a Heatmap Visual
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1
    )
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    print("$"*60)

# =====================================================================
# OBJECTIVE 1 (CONTINUED): BIVARIATE & MULTIVARIATE ANALYSIS
# =====================================================================
print("\n=== Processing Objective 1: Bivariate & Multivariate Analysis ===")

# Bivariate: Categorical vs Numerical (e.g., Customer Segment vs final_price)
if "customer_segment" in df.columns and "final_price" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="customer_segment",
        y="final_price",
        hue="customer_segment",
        estimator=np.mean,
        palette="pastel",
        legend=False,
    )
    plt.title("Bivariate Analysis: Average Spending by Customer Segment")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bivariate_segment_vs_spending.png")
    plt.close()

    print("#"*60)

# Multivariate Interaction: Product Category x Discount x Final Price
if (
    all(col in df.columns for col in ["product_category", "final_price"])
    and "discount_percentage" in df.columns
):
    plt.figure(figsize=(10, 6))
    # Create a scatter plot visualizing 3 variables simultaneously
    sns.scatterplot(
        data=df.sample(
            min(5000, len(df))
        ),  # Sample 5,000 points to keep plot fast/readable
        x="discount_percentage",
        y="final_price",
        hue="product_category",
        alpha=0.6,
    )
    plt.title(
        "Multivariate Analysis: Interaction of Discount, Final Price & Product Category"
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multivariate_interaction.png")
    plt.close()

print(
    f"\nEDA Framework Executed Successfully! All figures saved in the '{output_dir}' directory."
)



