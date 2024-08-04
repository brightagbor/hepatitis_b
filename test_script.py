# Open the existing requirements.txt file in read mode
with open('requirements.txt', 'r') as file:
    lines = file.readlines()

# Open the same file in write mode to overwrite it
with open('requirements.txt', 'w') as file:
    for line in lines:
        # Write only lines that do not contain '@'
        if '@' not in line:
            file.write(line)
