dictionary = {
    "gender": "Female",
    "sex": "male"
}

for key, value in dictionary.items():
    if value == "male":
        dictionary[key] = 1
    else:
        dictionary[key] = 0

print(dictionary)