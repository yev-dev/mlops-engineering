import pytest
from fastapi.testclient import TestClient
from typing import Union
from src.main import app


@pytest.fixture(scope="function")
def test_client():

    with TestClient(
        app,
        raise_server_exceptions=True,
        root_path="",
        follow_redirects=True
    ) as test_client:
        yield test_client

def test_root(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to the model service"



@pytest.mark.parametrize(
    "json_data, expected_status_code",
    [
        ({"data": [80]}, 200),  # Test with valid data
    ],
)
def test_single_prediction(
    test_client: TestClient, json_data: Union[dict, str], expected_status_code: int
) -> None:

    response = test_client.post(
        f"/predict",
        json=json_data,
    )

    assert response.status_code == expected_status_code
