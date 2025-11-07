#!/bin/bash

# Directories
ORIGINAL_DIR="../../puffer_data_original/testing_data"
CLEANED_DIR="../../puffer_data_cleaned/testing_data"
SCRIPT_PATH="data_processing_new.py"  # Adjust if your Python script has a different name

# Create output directory if it doesn't exist
mkdir -p "$CLEANED_DIR"

# Find all video_sent files and extract their date patterns
for sent_file in "$ORIGINAL_DIR"/video_sent_*.csv; do
    # Check if files exist (handles case where no matches found)
    if [ ! -f "$sent_file" ]; then
        echo "No video_sent files found in $ORIGINAL_DIR"
        exit 1
    fi
    
    # Extract the date pattern (everything after "video_sent_")
    basename_sent=$(basename "$sent_file")
    date_pattern="${basename_sent#video_sent_}"
    
    # Construct the corresponding acked file name
    acked_file="$ORIGINAL_DIR/video_acked_$date_pattern"
    
    # Check if corresponding acked file exists
    if [ ! -f "$acked_file" ]; then
        echo "⚠️  Warning: No matching acked file for $basename_sent"
        echo "   Expected: $acked_file"
        continue
    fi
    
    # Construct output file name
    output_file="$CLEANED_DIR/${date_pattern%.csv}_training_data.csv"
    
    # Process the pair
    echo "=========================================="
    echo "Processing: $date_pattern"
    echo "  Sent:   $sent_file"
    echo "  Acked:  $acked_file"
    echo "  Output: $output_file"
    echo "=========================================="
    
    # Run the Python script
    python3 "$SCRIPT_PATH" "$sent_file" "$acked_file" "$output_file"
    
    # Check if processing was successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed $date_pattern"
    else
        echo "❌ Error processing $date_pattern"
    fi
    echo ""
done

echo "=========================================="
echo "All files processed!"
echo "=========================================="