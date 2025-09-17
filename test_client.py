import requests

url = "http://127.0.0.1:8000/chat"

tests = [
    {
        "name": "Ping/Pong",
        "data": {"chat_id": "sanity-check-ping", "messages":[{"type":"text","content":"ping"}]},
        "expected": {"message":"pong","base_random_keys":None,"member_random_keys":None}
    },
    {
        "name": "Base Random Key",
        "data": {"chat_id": "sanity-check-base-key", "messages":[{"type":"text","content":"return base random key: 123e4567-e89b-12d3-a456-426614174000"}]},
        "expected": {"message":None,"base_random_keys":["123e4567-e89b-12d3-a456-426614174000"],"member_random_keys":None}
    },
    {
        "name": "Member Random Key",
        "data": {"chat_id": "sanity-check-member-key", "messages":[{"type":"text","content":"return member random key: abc-def-123"}]},
        "expected": {"message":None,"base_random_keys":None,"member_random_keys":["abc-def-123"]}
    },
    {
        "name": "Something",
        "data": {"chat_id": "sanity-check-member-key", "messages":[{"type":"text","content":"Tell me something"}]},
        "expected": {"message":None,"base_random_keys":None,"member_random_keys":["abc-def-123"]}
    }
]

for test in tests:
    resp = requests.post(url, json=test["data"], proxies={"http": None, "https": None})
    print(f"Test: {test['name']}")
    print("Response:", resp.json())
    print("Expected:", test["expected"])
    print("Pass:", resp.json() == test["expected"])
    print("-"*40)
