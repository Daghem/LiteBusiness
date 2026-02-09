import requests

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "content": "Qual è la capitale della Francia?"}
response = requests.post("http://localhost:8000/", headers=headers)
print(response.json())