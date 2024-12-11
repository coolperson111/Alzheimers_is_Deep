#!/bin/bash

# Directories
INPUT_DIR="/data/malhar/AD/temp_files"
OUTPUT_DIR="/data/malhar/AD/processed"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all .vcf.gz files in the input directory
for vcf_file in "$INPUT_DIR"/*.vcf.gz; do
    # Extract the base name of the file (e.g., chr2.vcf.gz -> chr2)
    base_name=$(basename "$vcf_file" .vcf.gz)

    # Prepare the output filtered VCF file path
    output_vcf="$OUTPUT_DIR/${base_name}.filtered.vcf.gz"

    # Run vcftools with the specified filters
    echo "Running vcftools on $vcf_file..."
    vcftools --gzvcf "$vcf_file" \
             --hwe 0.05 \
             --minGQ 20 \
             --maf 0.01 \
             --max-missing 0.95 \
             --recode --recode-INFO-all --out "${OUTPUT_DIR}/${base_name}.filtered"

    # Compress the filtered VCF file
    # mv "${OUTPUT_DIR}/${base_name}.filtered.recode.vcf" "$output_vcf"

    echo "Filtered VCF saved to $output_vcf"
done

echo "All files processed. Filtered VCFs are in $OUTPUT_DIR."
