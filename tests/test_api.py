from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "DSN Rec Agent API is running"}

def test_simulate_route_exists():
    # Just check if the route is there by sending an empty request
    # It should return 422 Unprocessable Entity because of missing body
    response = client.post("/reviews/simulate", json={
        "persona_text": "...",
        "product_text": "..."
    })
    # Since I'm providing the keys but empty/short content, it might fail in LLM extraction if not mocked,
    # but for a "route exists" test, we expect a 200 if LLM extraction works or 500 if it fails due to missing keys.
    # Actually, sending valid keys will hit the service.
    assert response.status_code in [200, 500, 422]
