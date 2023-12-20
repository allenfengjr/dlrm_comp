import re
# Read the log data from a file
log_file_path = "/N/u/haofeng/BigRed200/dlrm/cyclic_2098199.log"  # Replace with the actual file path
with open(log_file_path, "r") as file:
    log_data = file.read()

# Use regular expressions to extract test accuracy
pattern = r"Testing at - \d+/\d+ of epoch \d+,\n accuracy ([\d.]+) %, best [\d.]+ %"
accuracy_matches = re.findall(pattern, log_data)

# Print the test accuracy values
for accuracy in accuracy_matches:
    print("Test Accuracy:", accuracy)