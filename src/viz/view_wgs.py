import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cyvcf2 import VCF

# Path to your VCF file
vcf_file = "data/genetic/wgs/002_snps/002_S_0413_SNPs.vcf"

# Load VCF file
vcf = VCF(vcf_file)

# Load VCF into a DataFrame
data = []
for variant in vcf:
    #     print(
    #         f"Chromosome: {variant.CHROM}, Position: {variant.POS}, REF: {variant.REF}, ALT: {variant.ALT}"
    #     )
    data.append(
        {
            "CHROM": variant.CHROM,
            "POS": variant.POS,
            "REF": variant.REF,
            "ALT": ",".join(variant.ALT),
            "QUAL": variant.QUAL,
            "FILTER": variant.FILTER,
        }
    )

vcf_df = pd.DataFrame(data)
print(vcf_df.head())


# Count variants per chromosome
chrom_counts = vcf_df["CHROM"].value_counts()

# Bar plot
plt.figure(figsize=(10, 5))
sns.barplot(x=chrom_counts.index, y=chrom_counts.values, palette="viridis")
plt.title("Variants per Chromosome")
plt.xlabel("Chromosome")
plt.ylabel("Number of Variants")
plt.xticks(rotation=90)
plt.show()

# Quality score histogram
plt.figure(figsize=(8, 5))
sns.histplot(vcf_df["QUAL"], bins=50, kde=True, color="blue")
plt.title("Quality Score Distribution")
plt.xlabel("Quality Score")
plt.ylabel("Frequency")
plt.show()

def classify_variant(row):
    ref_len = len(row["REF"])
    alt_len = len(row["ALT"])
    if ref_len == 1 and alt_len == 1:
        return "SNP"
    elif ref_len > alt_len:
        return "Deletion"
    elif ref_len < alt_len:
        return "Insertion"
    else:
        return "Other"

vcf_df["Variant_Type"] = vcf_df.apply(classify_variant, axis=1)

# Count and plot
variant_type_counts = vcf_df["Variant_Type"].value_counts()

plt.figure(figsize=(8, 5))
sns.barplot(x=variant_type_counts.index, y=variant_type_counts.values, palette="coolwarm")
plt.title("Distribution of Variant Types")
plt.xlabel("Variant Type")
plt.ylabel("Count")
plt.show()


# Parse allele frequency from INFO field (example: AF=0.123)
def parse_af(info):
    for field in info.split(";"):
        if field.startswith("AF="):
            return float(field.split("=")[1])
    return None

# vcf_df["Allele_Frequency"] = vcf_df["INFO"].apply(parse_af)

# # Plot allele frequency distribution
# plt.figure(figsize=(8, 5))
# sns.histplot(vcf_df["Allele_Frequency"].dropna(), bins=50, kde=True, color="green")
# plt.title("Allele Frequency Distribution")
# plt.xlabel("Allele Frequency")
# plt.ylabel("Frequency")
# plt.show()

# Density plot of variants across positions
plt.figure(figsize=(10, 5))
sns.scatterplot(data=vcf_df, x="POS", y="CHROM", s=10, alpha=0.5)
plt.title("Genome-Wide Variant Density")
plt.xlabel("Position")
plt.ylabel("Chromosome")
plt.show()
