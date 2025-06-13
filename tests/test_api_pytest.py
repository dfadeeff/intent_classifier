#!/usr/bin/env python3
"""
Simple API tests - converted from simple_api_test_manual.py
Very basic tests that are forgiving about implementation details
"""

import pytest
import requests
from requests.exceptions import ConnectionError

# Test configuration
BASE_URL = "http://127.0.0.1:8080"


def check_server_running():
    """Check if the server is running before running tests"""
    try:
        response = requests.get(f"{BASE_URL}/ready", timeout=2)
        return True
    except (ConnectionError, requests.exceptions.Timeout):
        return False


@pytest.fixture(scope="session", autouse=True)
def ensure_server_running():
    """Ensure server is running before any tests run"""
    if not check_server_running():
        pytest.skip(
            f"Server not running at {BASE_URL}. Start the server first with: python server.py"
        )


class TestAPI:
    """Simple API tests"""

    def test_health_check(self):
        """Test /ready endpoint"""
        response = requests.get(f"{BASE_URL}/ready")
        assert response.status_code == 200
        assert response.text == "OK"

    def test_intent_classification_success(self):
        """Test successful intent classification"""
        test_cases = [
            "find me a flight to boston",
            "what is the cheapest fare",
            "which airlines fly to denver",
            "what airports are in chicago",
        ]

        for text in test_cases:
            payload = {"text": text}
            response = requests.post(f"{BASE_URL}/intent", json=payload)

            assert response.status_code == 200

            result = response.json()
            assert "intents" in result
            assert len(result["intents"]) <= 3  # Top 3 max

            # Check each prediction
            for intent in result["intents"]:
                assert "label" in intent
                assert "confidence" in intent
                assert isinstance(intent["label"], str)
                assert isinstance(intent["confidence"], float)
                assert 0 <= intent["confidence"] <= 1

    def test_empty_text_error(self):
        """Test empty text returns 400"""
        payload = {"text": ""}
        response = requests.post(f"{BASE_URL}/intent", json=payload)

        assert response.status_code == 400
        result = response.json()
        assert result["label"] == "TEXT_EMPTY"
        assert "empty" in result["message"].lower()

    def test_whitespace_text_error(self):
        """Test whitespace-only text returns 400"""
        payload = {"text": "   "}
        response = requests.post(f"{BASE_URL}/intent", json=payload)

        assert response.status_code == 400
        result = response.json()
        assert result["label"] == "TEXT_EMPTY"

    def test_missing_text_field_error(self):
        """Test missing text field returns 400"""
        payload = {}
        response = requests.post(f"{BASE_URL}/intent", json=payload)

        assert response.status_code == 400
        result = response.json()
        assert result["label"] == "MISSING_TEXT_FIELD"

    def test_invalid_content_type_error(self):
        """Test invalid content type returns 400"""
        response = requests.post(
            f"{BASE_URL}/intent",
            data="not json",
            headers={"Content-Type": "text/plain"},
        )

        assert response.status_code == 400
        result = response.json()
        assert result["label"] == "INVALID_CONTENT_TYPE"

    def test_invalid_json_error(self):
        """Test invalid JSON returns error (400 or 500 both acceptable)"""
        response = requests.post(
            f"{BASE_URL}/intent",
            data="invalid json{",
            headers={"Content-Type": "application/json"},
        )

        # Server should return an error (400 for proper handling, 500 if not handled gracefully)
        assert response.status_code >= 400
        # We don't check the response body since it might not be JSON

    def test_invalid_endpoint_error(self):
        """Test invalid endpoint returns 404"""
        response = requests.get(f"{BASE_URL}/invalid")

        assert response.status_code == 404
        result = response.json()
        assert result["label"] == "NOT_FOUND"

    def test_wrong_method_error(self):
        """Test wrong HTTP method returns 405"""
        response = requests.get(f"{BASE_URL}/intent")

        assert response.status_code == 405
        result = response.json()
        assert result["label"] == "METHOD_NOT_ALLOWED"


# Run with: pytest tests/test_api.py -v
if __name__ == "__main__":
    print("Run with: pytest tests/test_api.py -v")
