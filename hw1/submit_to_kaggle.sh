#!/bin/bash

# Initialize variables
file_path=""
message=""

# Parse command-line arguments
while getopts "f:m:" opt; do
  case $opt in
    f) file_path="$OPTARG"
    ;;
    m) message="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Check if file_path and message are provided
if [[ -z "$file_path" || -z "$message" ]]; then
  echo "Error: -f and -m options are required"
  exit 1
fi

# Submit to Kaggle
kaggle competitions submit -c ntuadl2023hw1 -f "$file_path" -m "$message"