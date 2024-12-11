import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load data
file_path = "data/genetic/gwas/002_S_0295.csv"
gwas_df = pd.read_csv(file_path)

# Summary statistics
print(gwas_df.describe())

# GC Score distribution
sns.histplot(gwas_df["GC Score"], bins=50, kde=True, color="blue")
plt.title("GC Score Distribution")
plt.xlabel("GC Score")
plt.ylabel("Frequency")
plt.show()

# GT Score distribution
sns.histplot(gwas_df["GT Score"], bins=50, kde=True, color="green")
plt.title("GT Score Distribution")
plt.xlabel("GT Score")
plt.ylabel("Frequency")
plt.show()

# Count variants per chromosome
chrom_counts = gwas_df["Chr"].value_counts()

# Bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=chrom_counts.index.astype(str), y=chrom_counts.values, palette="viridis")
plt.title("Variants per Chromosome")
plt.xlabel("Chromosome")
plt.ylabel("Number of Variants")
plt.xticks(rotation=90)
plt.show()

# B Allele Frequency histogram
sns.histplot(gwas_df["B Allele Freq"], bins=50, kde=True, color="purple")
plt.title("B Allele Frequency Distribution")
plt.xlabel("Frequency")
plt.ylabel("Frequency")
plt.show()

# Log R Ratio histogram
sns.histplot(gwas_df["Log R Ratio"], bins=50, kde=True, color="orange")
plt.title("Log R Ratio Distribution")
plt.xlabel("Log R Ratio")
plt.ylabel("Frequency")
plt.show()
