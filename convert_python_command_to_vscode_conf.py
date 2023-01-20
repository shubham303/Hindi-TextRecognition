import json

# Store the Python command in a variable
python_command = "python create_lmdb_dataset.py Hindi/ Hindi/test.tsv lmdb True 5000"

# Split the command into a list of individual words
command_parts = python_command.split()

# Get the path to the Python executable
python_executable = command_parts[0]

# Get the path to the script
script_path = command_parts[1]

# Create the configuration object
config = {
    "name": "My Python Script",
    "type": "python",
    "request": "launch",
    "program": script_path,
    "console": "integratedTerminal",
    "env": {},
    "args": [],
    "cwd": "${workspaceFolder}",
    "envFile": "${workspaceFolder}/.env",
    "pythonPath": python_executable,
    "debugOptions": [
        "RedirectOutput"
    ]
}

# Add any additional arguments to the args list
for i in range(2, len(command_parts)):
    config["args"].append(command_parts[i])

# Print the configuration object as a JSON string
print(json.dumps(config ,  indent=2))
