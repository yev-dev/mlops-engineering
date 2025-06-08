#!/bin/bash

# This script simulates sending 1000 requests to a prediction service
# with random data for wine quality prediction.
# Ensure the service is running before executing this script.
# Usage: ./simulate_requests.sh


MODEL_SERVICE_PORT=5000

for i in {1..1000}
do
  # Generate random pricing data with a range of 80 to 85
  # This simulates stock prices for a prediction service
  stock_prices=$(awk -v min=80 -v max=85 'BEGIN{srand(); print min+rand()*(max-min)}')

  # Construct the JSON payload
json_data="{\"data\": [$stock_prices]}"

# Send the curl request
curl -X POST "http://localhost:${MODEL_SERVICE_PORT}/predict" \
      -H "Content-Type: application/json" \
      -d "$json_data"

echo "Request $i sent"

  # Optional: add a small delay to avoid overwhelming the server
  #sleep 0.1
done