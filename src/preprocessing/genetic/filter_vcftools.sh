#!/bin/bash

# Directory to store the processed files
PROCESSED_DIR="./data/genetic/wgs"

# Create the processed directory if it doesn't exist
mkdir -p $PROCESSED_DIR

# Loop through all the tar files
for tar_file in ./data/genetic/wgs/*chr2*.tar; do
    # Extract the tar files to a temporary directory
    echo "Extracting $tar_file..."
    mkdir -p temp_dir
    tar -xf "$tar_file" -C temp_dir

    # Get the gzipped VCF file name (assumes the tar file contains a single gzipped VCF file)
    vcf_file=$(ls temp_dir/*.vcf.gz)
    
    if [[ -f "$vcf_file" ]]; then
        # Prepare the output filtered VCF file
        output_vcf="$PROCESSED_DIR/$(basename "$tar_file" .tar).filtered.vcf.gz"
        
        # Run vcftools with the specified filters
        echo "Running vcftools on $vcf_file..."

        vcftools --gzvcf "$vcf_file" \
                 --hwe 0.05 \
                 --minGQ 20 \
                 --maf 0.01 \
                 --max-missing 0.95 \
                 --recode --recode-INFO-all --out "$output_vcf"

        # Move the filtered gzipped VCF to the processed directory
        mv "$output_vcf.recode.vcf.gz" "$PROCESSED_DIR/$(basename "$tar_file" .tar).filtered.vcf.gz"

        # Clean up the temporary directory
        rm -r temp_dir
    else
        echo "No gzipped VCF file found in $tar_file, skipping..."
    fi
done

echo "Filtering completed. All filtered VCF files are in the '$PROCESSED_DIR' directory."

