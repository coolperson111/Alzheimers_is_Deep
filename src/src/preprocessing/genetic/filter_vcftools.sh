#!/bin/bash

# Directory to store the processed files
PROCESSED_DIR="/data/malhar/AD/processed"

# Create the processed directory if it doesn't exist
mkdir -p $PROCESSED_DIR

# Loop through all the tar files for chr1 to chr9
for tar_file in /data/malhar/AD/*chr[1-9].vcf.tar; do
    # Extract the tar files to a temporary directory
    echo "Extracting $tar_file..."
    # mkdir -p temp_dir
    tar -xf "$tar_file" -C temp_dir

    # Get the gzipped VCF file name (assumes the tar file contains a single gzipped VCF file)
    vcf_file=$(ls temp_dir/*.vcf.gz)
    
    if [[ -f "$vcf_file" ]]; then
        # Extract only the base name of the tar file (e.g., chr1.vcf.tar -> chr1.vcf)
        base_name=$(basename "$tar_file" .vcf.tar)

        # Prepare the output filtered VCF file
        output_vcf="$PROCESSED_DIR/${base_name}.filtered.vcf.gz"
        
        # Run vcftools with the specified filters
        echo "Running vcftools on $vcf_file..."

        vcftools --gzvcf "$vcf_file" \
                 --hwe 0.05 \
                 --minGQ 20 \
                 --maf 0.01 \
                 --max-missing 0.95 \
                 --recode --recode-INFO-all --out "${PROCESSED_DIR}/${base_name}.filtered"

        # Compress the output VCF file and clean up
        mv "${PROCESSED_DIR}/${base_name}.filtered.recode.vcf" "$output_vcf"

        # Clean up the temporary directory
        # rm -r temp_dir
    else
        echo "No gzipped VCF file found in $tar_file, skipping..."
    fi
done

echo "Filtering completed. All filtered VCF files are in the '$PROCESSED_DIR' directory."
