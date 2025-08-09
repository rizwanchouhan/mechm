import json  # Module for working with JSON data
import csv  # Module for working with CSV files

# specify input and output file paths
input_file = 'ccs_synthetic_filtered_large.json'  # Path to input JSON file
output_file = 'ccs_synthetic_filtered_large.tsv'  # Path to output TSV file

# load JSON data from input file
with open(input_file, 'r') as f:
    data = json.load(f)  # Load JSON data

# extract header and data from JSON
header = data[0].keys()  # Extract header from JSON data
rows = [x.values() for x in data]  # Extract rows from JSON data

# write data to TSV file
with open(output_file, 'w') as f:
    writer = csv.writer(f, delimiter='\t')  # Initialize CSV writer with tab delimiter
    writer.writerow(header)  # Write header to TSV file
    writer.writerows(rows)  # Write rows to TSV file
