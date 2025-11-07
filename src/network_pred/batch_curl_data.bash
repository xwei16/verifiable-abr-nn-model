#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_date> <end_date>"
    echo "Example: $0 2025-11-01 2025-11-04"
    echo ""
    echo "Date format: YYYY-MM-DD"
    exit 1
fi

START_DATE="$1"
END_DATE="$2"
OUTPUT_DIR="../../puffer_data_original/testing_data"
BASE_URL="https://storage.googleapis.com/puffer-data-release"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Validate date format (convert to seconds since epoch for comparison)
START_SEC=$(date -d "$START_DATE" "+%s" 2>/dev/null)
END_SEC=$(date -d "$END_DATE" "+%s" 2>/dev/null)

if [ -z "$START_SEC" ]; then
    echo "❌ Error: Invalid start date format. Use YYYY-MM-DD"
    exit 1
fi

if [ -z "$END_SEC" ]; then
    echo "❌ Error: Invalid end date format. Use YYYY-MM-DD"
    exit 1
fi

# Check if end date is after start date
if [ "$END_SEC" -lt "$START_SEC" ]; then
    echo "❌ Error: End date must be after or equal to start date"
    exit 1
fi

echo "=========================================="
echo "Downloading Puffer data files"
echo "Date range: $START_DATE to $END_DATE"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Function to download a file with error checking
download_file() {
    local url="$1"
    local output_path="$2"
    local filename=$(basename "$output_path")
    
    echo "⬇️  Downloading: $filename"
    echo "URL: $url"
    echo $output_path
    if curl -f -L -o "$output_path" "$url" 2>/dev/null; then
        echo "✅ Success: $filename"
        return 0
    else
        echo "❌ Failed: $filename (file may not exist)"
        # Remove partial download if it exists
        rm -f "$output_path"
        return 1
    fi
}


# Initialize counters
total_attempts=0
successful_downloads=0

# Loop through each date in the range
current_date="$START_DATE"
current_sec="$START_SEC"

while [ "$current_sec" -le "$END_SEC" ]; do
    # Calculate next date (current_date + 1 day at 11:00)
    next_date=$(date -d "$current_date + 1 day" "+%Y-%m-%d")
    
    # Format: 2025-11-04T11_2025-11-05T11
    date_pattern="${current_date}T11_${next_date}T11"
    
    echo "------------------------------------------"
    echo "Processing: $date_pattern"
    echo "------------------------------------------"
    
    # Download video_sent file
    sent_filename="video_sent_${date_pattern}.csv"
    sent_url="$BASE_URL/${date_pattern}/${sent_filename}"
    sent_output="$OUTPUT_DIR/$sent_filename"
    
    total_attempts=$((total_attempts + 1))
    if download_file "$sent_url" "$sent_output"; then
        successful_downloads=$((successful_downloads + 1))
    fi
    echo ""
    
    # Download video_acked file
    acked_filename="video_acked_${date_pattern}.csv"
    acked_url="$BASE_URL/${date_pattern}/${acked_filename}"
    acked_output="$OUTPUT_DIR/$acked_filename"
    
    total_attempts=$((total_attempts + 1))
    if download_file "$acked_url" "$acked_output"; then
        successful_downloads=$((successful_downloads + 1))
    fi
    echo ""
    
    # Move to next date
    current_date="$next_date"
    current_sec=$(date -d "$current_date" "+%s")
done

echo "=========================================="
echo "Download Summary"
echo "=========================================="
echo "Total files attempted: $total_attempts"
echo "Successfully downloaded: $successful_downloads"
echo "Failed: $((total_attempts - successful_downloads))"
echo ""
echo "Files saved to: $OUTPUT_DIR"
echo "=========================================="