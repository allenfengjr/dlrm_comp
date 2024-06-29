import re

# Path to the log file to read from
log_file_path_to_read = "/root/dlrm_comp/output.log"
# Path to the new file to write the parsed accuracy
log_file_path_to_write = "/root/dlrm_comp/accuracy.log"

# Read the content of the log file
with open(log_file_path_to_read, "r") as file:
    log_text = file.read()

# Regular expression to find the accuracy percentages
pattern = r"accuracy ([\d.]+) %"

matches = re.findall(pattern, log_text)

# Prepare the accuracy to be written to a new file
content_to_write = "\n".join(matches)

# Write to the new file
with open(log_file_path_to_write, "w") as file:
    file.write(content_to_write)