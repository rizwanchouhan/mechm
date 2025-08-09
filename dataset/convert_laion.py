import json  # Module for working with JSON data
import csv  # Module for working with CSV files

# specify input and output file paths
input_file = 'laion_synthetic_filtered_large.json'  # Path to the input JSON file
output_file = 'laion_synthetic_filtered_large.tsv'  # Path to the output TSV file

# load JSON data from input file
with open(input_file, 'r') as f:
    data = json.load(f)  # Load JSON data from the input file

# extract header and data from JSON
header = data[0].keys()  # Extract the header from the JSON data
rows = [x.values() for x in data]  # Extract rows of data from the JSON

# write data to TSV file
with open(output_file, 'w') as f:
    writer = csv.writer(f, delimiter='\t')  # Initialize CSV writer with tab delimiter
    writer.writerow(header)  # Write the header to the TSV file
    writer.writerows(rows)  # Write the rows of data to the TSV file
