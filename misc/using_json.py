import json

# writing to a json file
with open("data_filename.json","w") as write_file:
    json.dump(data,write_file) #dumps takes arguments, 1, data, and 2, filename to be written to

# writing to a string:
json_string = json.dumps(data)

# Unserializing

unencoded_data = json.loads(encoded_data)

# or

with open("data_file.json","r") as read_file:
    data = json.load(read_file)

# Example json string
json_string = """
{
    "researcher": {
        "name": "Ford Prefect",
        "species": "Betelgeusian",
        "relatives": [
            {
                "name": "Zaphod Beeblebrox",
                "species": "Betelgeusian"
            }
        ]
    }
}
"""
data = json.loads(json_string)

# example with requests
response = requests.get("https://jsonplaceholder.typicode.com/todos")
todos = json.loads(response.text)