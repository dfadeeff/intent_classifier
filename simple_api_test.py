#!/usr/bin/env python3
import json

import requests

base_url = "http://127.0.0.1:8080"


def print_json_response(response):
    """Pretty print JSON response"""
    try:
        json_data = response.json()
        print(f"Status: {response.status_code}")
        print(f"JSON Response:")
        print(json.dumps(json_data, indent=2))
    except:
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")


print("🔍 Testing health check...")
response = requests.get(f"{base_url}/ready")
print(f"Health check: {response.status_code} - {response.text}")

print("\n" + "=" * 60)
print("🎯 TESTING INTENT CLASSIFICATION - FULL JSON RESPONSES")
print("=" * 60)

test_cases = [
    "find me a flight to boston",
    "what is the cheapest fare",
    "which airlines fly to denver",
    "what airports are in chicago",
]

for i, text in enumerate(test_cases, 1):
    print(f"\n{i}. Testing: '{text}'")
    print("-" * 50)

    payload = {"text": text}
    response = requests.post(f"{base_url}/intent", json=payload)
    print_json_response(response)

    # Show all 3 predictions if available
    if response.status_code == 200:
        result = response.json()
        print(f"\nTop predictions:")
        for j, intent in enumerate(result["intents"], 1):
            print(f"  {j}. {intent['label']} (confidence: {intent['confidence']})")

print("\n" + "=" * 60)
print("🚨 TESTING ERROR CASES")
print("=" * 60)

# Test 1: Empty text (should return 400 with TEXT_EMPTY)
print("\n1. Testing empty text:")
print("-" * 30)
payload = {"text": ""}
response = requests.post(f"{base_url}/intent", json=payload)
print_json_response(response)

# Test 2: Whitespace only text (should return 400 with TEXT_EMPTY)
print("\n2. Testing whitespace-only text:")
print("-" * 30)
payload = {"text": "   "}
response = requests.post(f"{base_url}/intent", json=payload)
print_json_response(response)

# Test 3: Missing text field (should return 400 with MISSING_TEXT_FIELD)
print("\n3. Testing missing text field:")
print("-" * 30)
payload = {}
response = requests.post(f"{base_url}/intent", json=payload)
print_json_response(response)

# Test 4: Invalid content type (should return 400 with INVALID_CONTENT_TYPE)
print("\n4. Testing invalid content type:")
print("-" * 30)
response = requests.post(
    f"{base_url}/intent", data="not json", headers={"Content-Type": "text/plain"}
)
print_json_response(response)

# Test 5: Invalid JSON
print("\n5. Testing invalid JSON:")
print("-" * 30)
response = requests.post(
    f"{base_url}/intent",
    data="invalid json{",
    headers={"Content-Type": "application/json"},
)
print_json_response(response)

# Test 6: Invalid endpoint (should return 404 with NOT_FOUND)
print("\n6. Testing invalid endpoint:")
print("-" * 30)
response = requests.get(f"{base_url}/invalid")
print_json_response(response)

# Test 7: Wrong HTTP method (should return 405 with METHOD_NOT_ALLOWED)
print("\n7. Testing wrong HTTP method:")
print("-" * 30)
response = requests.get(f"{base_url}/intent")
print_json_response(response)

print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)
print("✅ All tests completed!")
print("Check that:")
print("• Intent responses have 'intents' array with top 3 predictions")
print("• Each prediction has 'label' (string) and 'confidence' (float)")
print("• Error cases return proper HTTP codes and error messages")
print("• Health check returns 200 OK")
