import json

def ThemeValidation(theme: str):
    json_data = json.dumps({"message": f"Theme '{theme}' is under development."})
    return json_data