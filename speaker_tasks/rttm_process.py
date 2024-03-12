import json
import glob

rttm_files = glob.glob('/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/oracle_vad/denoised_crowd_test_rttms/*.rttm')
output_file = 'denoised_predicted.json'

json_data = {}  # JSON object to store the results

for rttm_file in rttm_files:
    
    with open(rttm_file, 'r') as file:
        lines = file.readlines()
        
    # Extract unique speaker IDs
    unique_speakers = set()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 8:
            file_name = parts[1]
            speaker_id = parts[7]
            unique_speakers.add(speaker_id)

    # Add the result to the JSON object
    json_data[file_name] = len(unique_speakers)

# Write JSON to file
with open(output_file, 'w') as file:
    json.dump(json_data, file, indent=4)
