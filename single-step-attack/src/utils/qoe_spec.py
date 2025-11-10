import json
import numpy as np

SPEC_FILE = "pensieve_spectra_data/useful_results_from_spectra/abr_original_fullspec_5float.txt"
SPEC_JSON_OUT_FILE = "data/full_spec.json"
brs = [300, 750, 1200, 1850, 2850, 4300]

def compute_qoe(last_bitrates_l, last_bitrates_u, current_bitrates):
    qoe_set = []

    # get the last bitrates
    l = int(last_bitrates_l*4300)
    u = int(last_bitrates_u*4300)
    last_bitrates = [br for br in brs if l <= br <= u]
    
    # get qoe
    for last in last_bitrates:
        for curr in current_bitrates:
            last_q = np.log(last / brs[0])
            cur_q = np.log(curr / brs[0])
            qoe_2 =  last_q + cur_q - np.abs(last_q - cur_q)
            qoe_set += [round(qoe_2,5)]
    qoe_set.sort(reverse=True)
    return qoe_set
    
# Read your input file
with open(SPEC_FILE, "r") as file:
    content = file.read()

# Split blocks by the separator line
blocks = [block.strip() for block in content.split("--------------------------------------------------") if block.strip()]

# Prepare list of parsed JSON objects
parsed_data = []

for block in blocks:
    lines = block.splitlines()
    json_data = json.loads(lines[0])
    output_line = lines[1].strip()
    
    # Parse outputs as a list of floats
    outputs = output_line.replace("output:", "").strip().strip("{}").split(",")
    outputs = [float(o.strip()) for o in outputs if o.strip()]
    
    # Flatten the json_data values
    flattened = {}
    for key, values in json_data.items():
        if key == "Last1_chunk_bitrate":
            l = int(values[0]*4300)
            u = int(values[1]*4300)
            last_bitrates = [br for br in brs if l <= br <= u]
            flattened["Last1_chunk_bitrate_l"] = last_bitrates[0] / 4300
            flattened["Last1_chunk_bitrate_u"] = last_bitrates[0] / 4300   # we assume that there's only one last bitrate
            continue
        for i, val in enumerate(values):
            if i == 0:
                flattened[f"{key}_l"] = val
            else:
                flattened[f"{key}_u"] = val

    outputs.sort()
    flattened["cur_br"] = outputs

    flattened["qoe_2"] = compute_qoe(flattened["Last1_chunk_bitrate_l"], flattened["Last1_chunk_bitrate_u"], outputs)
    flattened["qoe_2"].sort()
    parsed_data.append(flattened)

# Write to a JSON file
with open(SPEC_JSON_OUT_FILE, "w") as jsonfile:
    json.dump(parsed_data, jsonfile, indent=2)
