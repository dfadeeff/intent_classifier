#!/usr/bin/env python3
"""
Basic server unit tests
"""
import pytest
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server


def test_ready_endpoint_model_ready():
    """Test /ready when model is ready"""
    with server.app.test_client() as client:
        with patch.object(server.model, "is_ready", return_value=True):
            response = client.get("/ready")
            assert response.status_code == 200
            assert response.data.decode() == "OK"


def test_ready_endpoint_model_not_ready():
    """Test /ready when model is not ready"""
    with server.app.test_client() as client:
        with patch.object(server.model, "is_ready", return_value=False):
            response = client.get("/ready")
            assert response.status_code == 423


def test_404_handler():
    """Test 404 error handler"""
    with server.app.test_client() as client:
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.get_json()
        assert data["label"] == "NOT_FOUND"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
