#!/bin/bash

# This script simulates sending 1000 requests to a prediction service
# with random data for wine quality prediction.
# Ensure the service is running before executing this script.
# Usage: ./simulate_requests.sh


for i in {1..1000}
do
  # Generate random data
  stock_prices=$(awk -v min=80 -v max=85 'BEGIN{srand(); print min+rand()*(max-min)}')

  # Construct the JSON payload
json_data="{\"data\": [$stock_prices]}"

# Send the curl request
curl -X POST "http://127.0.0.1:8000/predict" \
      -H "Content-Type: application/json" \
      -d "$json_data"

echo "Request $i sent"

  # Optional: add a small delay to avoid overwhelming the server
  #sleep 0.1
done