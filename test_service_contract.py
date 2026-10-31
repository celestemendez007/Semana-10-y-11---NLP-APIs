"""
test_service_contract.py
Verifica app.py contra public_sample.json y valida el contrato OpenAPI
"""
import json, os, time
from fastapi.testclient import TestClient

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Configurar token de entorno antes de importar app
os.environ["QA_API_TOKEN"] = "dev_token_secret_123"
os.environ["MODEL_DIR"] = os.path.join(SCRIPT_DIR, "modernbert-base-squad2-final")

from app import app

def run_tests():
    print("=" * 60)
    print("VERIFICANDO CONTRATO DE SERVICIO API")
    print("=" * 60)

    with TestClient(app) as client:
        # 1. GET /health
        print("\n1. Test GET /health (public)...")
        r = client.get("/health")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        assert r.json() == {"status": "ok"}, f"Invalid health payload: {r.json()}"
        print("   [PASS] /health -> 200 {'status': 'ok'}")

        # 2. GET /info - unauthorized (no token)
        print("\n2. Test GET /info - Sin token (esperado 401)...")
        r = client.get("/info")
        assert r.status_code == 401, f"Expected 401, got {r.status_code}"
        assert r.json().get("error") == "unauthorized", f"Invalid error body: {r.json()}"
        print("   [PASS] /info sin token -> 401 {'error': 'unauthorized'}")

        # 3. GET /info - unauthorized (token inválido)
        print("\n3. Test GET /info - Token incorrecto (esperado 401)...")
        r = client.get("/info", headers={"Authorization": "Bearer wrong_token"})
        assert r.status_code == 401, f"Expected 401, got {r.status_code}"
        assert r.json().get("error") == "unauthorized"
        print("   [PASS] /info token invalido -> 401 {'error': 'unauthorized'}")

        # 4. GET /info - authorized
        print("\n4. Test GET /info - Token valido (esperado 200)...")
        r = client.get("/info", headers={"Authorization": "Bearer dev_token_secret_123"})
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        info_data = r.json()
        for field in ["team", "model", "params"]:
            assert field in info_data, f"Missing required field {field}"
        print(f"   [PASS] /info -> 200: model={info_data['model']}, params={info_data['params']:,}")

        # 5. POST /predict - public_sample.json
        print("\n5. Test POST /predict con public_sample.json...")
        with open("public_sample.json", "r", encoding="utf-8") as f:
            sample_data = json.load(f)

        headers = {
            "Authorization": "Bearer dev_token_secret_123",
            "Content-Type": "application/json",
        }

        results = []
        for item in sample_data["items"]:
            t0 = time.time()
            payload = {
                "id": item["id"],
                "context": item["context"],
                "question": item["question"],
            }
            r = client.post("/predict", json=payload, headers=headers)
            elapsed = (time.time() - t0) * 1000
            assert r.status_code == 200, f"Failed on {item['id']}: {r.status_code} {r.text}"
            res = r.json()

            # Contract checks
            assert res["id"] == item["id"], f"ID mismatch: {res['id']} vs {item['id']}"
            assert "answer" in res and "start" in res and "score" in res, f"Missing fields in {res}"
            if res["answer"] is None:
                assert res["start"] is None, f"start must be null when answer is null"
            else:
                assert res["start"] is not None, f"start cannot be null when answer is present"
                # Offset validity check (Section VII.4)
                expected_slice = item["context"][res["start"] : res["start"] + len(res["answer"])]
                assert expected_slice == res["answer"], (
                    f"Offset mismatch on {item['id']}: context[{res['start']}:{res['start']+len(res['answer'])}] "
                    f"= '{expected_slice}' vs answer '{res['answer']}'"
                )

            is_impossible = item["is_impossible"]
            ref_answers = [a["text"] for a in item.get("answers", [])]

            correct = False
            if is_impossible:
                correct = (res["answer"] is None)
            else:
                correct = (res["answer"] is not None) and (
                    (res["answer"] in ref_answers) or any(
                        res["answer"].lower() in a.lower() or a.lower() in res["answer"].lower()
                        for a in ref_answers
                    )
                )

            results.append({
                "id": item["id"],
                "answer": res["answer"],
                "start": res["start"],
                "score": res["score"],
                "latency_ms": round(elapsed, 1),
                "correct": correct,
                "expected": "null" if is_impossible else ref_answers,
            })
            print(f"   [{'PASS' if correct else 'FAIL'}] {item['id']} ({elapsed:.1f}ms): answer={repr(res['answer'])}, start={res['start']}, score={res['score']}")

        print("\n" + "=" * 60)
        print("RESUMEN DE PRUEBAS DEL SERVICIO")
        print("=" * 60)
        for res in results:
            print(f"  {res['id']}: Answer='{res['answer']}' (start={res['start']}) | Correct={res['correct']} | Latency={res['latency_ms']}ms")

if __name__ == "__main__":
    run_tests()
