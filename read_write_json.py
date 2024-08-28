import json
import os

def read_json(file_path, variable_name="image_data"):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return {}
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            json_data = content.split(f'const {variable_name} = ')[1].rstrip(';')
            data = json.loads(json_data)
        return data
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return {}

def write_json(file_path, data, variable_name="image_data"):
    try:
        json_data = json.dumps(data, indent=4)
        js_content = f"const {variable_name} = {json_data};"
        with open(file_path, 'w') as file:
            file.write(js_content)
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")
