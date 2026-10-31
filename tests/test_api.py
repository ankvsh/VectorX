import unittest

from vectordb.api import Services, create_app
from vectordb.api.app import build_services
from vectordb.config import Settings
from vectordb.services import OllamaError

VEC = ",".join(["0.9"] * 4 + ["0.1"] * 12)


class FakeOllama:
    embed_model = "fake-embed"
    gen_model = "fake-gen"

    def __init__(self, up: bool = True):
        self.up = up

    def is_available(self):
        return self.up

    def embed(self, text):
        if not self.up:
            raise OllamaError("Ollama unavailable")
        # Deterministic bag-of-letters embedding: similar texts -> similar vectors.
        vec = [0.0] * 26
        for ch in text.lower():
            if "a" <= ch <= "z":
                vec[ord(ch) - 97] += 1.0
        return vec

    def generate(self, prompt):
        if not self.up:
            raise OllamaError("Ollama unavailable")
        return "fake answer"


def make_client(up: bool = True):
    base = build_services(Settings())
    services = Services(base.settings, base.vector_db, base.doc_db, FakeOllama(up))
    return create_app(services).test_client()


class VectorEndpointsTest(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    def test_search_all_algorithms_agree_on_top_hit(self):
        tops = set()
        for algo in ("bruteforce", "kdtree", "hnsw"):
            r = self.client.get(f"/search?v={VEC}&k=3&metric=euclidean&algo={algo}")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(len(r.json["results"]), 3)
            tops.add(r.json["results"][0]["id"])
        self.assertEqual(len(tops), 1)

    def test_search_validation(self):
        self.assertEqual(self.client.get("/search?v=1,2,3").status_code, 400)
        self.assertEqual(self.client.get(f"/search?v={VEC}&metric=nope").status_code, 400)
        self.assertEqual(self.client.get(f"/search?v={VEC}&algo=nope").status_code, 400)

    def test_insert_and_delete(self):
        r = self.client.post(
            "/insert", json={"metadata": "new", "category": "x", "embedding": [0.5] * 16}
        )
        new_id = r.json["id"]
        self.assertEqual(len(self.client.get("/items").json), 21)
        self.assertTrue(self.client.delete(f"/delete/{new_id}").json["ok"])
        self.assertFalse(self.client.delete(f"/delete/{new_id}").json["ok"])
        self.assertEqual(len(self.client.get("/items").json), 20)

    def test_insert_rejects_bad_body(self):
        r = self.client.post("/insert", json={"metadata": "x", "embedding": [1, 2]})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.json)

    def test_benchmark_and_info(self):
        b = self.client.get(f"/benchmark?v={VEC}").json
        self.assertEqual(b["itemCount"], 20)
        info = self.client.get("/hnsw-info").json
        self.assertEqual(info["nodeCount"], 20)
        self.assertEqual(info["nodesPerLayer"][0], 20)

    def test_index_page_and_cors(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"<html", r.data.lower())
        self.assertEqual(r.headers["Access-Control-Allow-Origin"], "*")


class DocumentEndpointsTest(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    def test_rag_roundtrip(self):
        text = "dynamic programming stores subproblem results " * 5
        r = self.client.post("/doc/insert", json={"title": "DP", "text": text})
        self.assertEqual(r.json["chunks"], 1)
        self.assertEqual(r.json["dims"], 26)
        self.assertEqual(len(self.client.get("/doc/list").json), 1)

        r = self.client.post("/doc/ask", json={"question": "dynamic programming", "k": 3})
        self.assertEqual(r.json["answer"], "fake answer")
        self.assertEqual(r.json["contexts"][0]["title"], "DP")

        r = self.client.post("/doc/search", json={"question": "dynamic programming"})
        self.assertEqual(len(r.json["contexts"]), 1)

        doc_id = r.json["contexts"][0]["id"]
        self.assertTrue(self.client.delete(f"/doc/delete/{doc_id}").json["ok"])

    def test_long_document_is_chunked(self):
        text = " ".join(f"word{i}" for i in range(600))
        r = self.client.post("/doc/insert", json={"title": "Long", "text": text})
        self.assertGreater(r.json["chunks"], 1)
        titles = [d["title"] for d in self.client.get("/doc/list").json]
        self.assertTrue(titles[0].startswith("Long [1/"))

    def test_validation(self):
        self.assertEqual(self.client.post("/doc/insert", json={"title": "x"}).status_code, 400)
        self.assertEqual(self.client.post("/doc/ask", json={}).status_code, 400)

    def test_ollama_offline(self):
        client = make_client(up=False)
        r = client.post("/doc/insert", json={"title": "t", "text": "some text"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("error", r.json)
        self.assertEqual(client.get("/doc/list").json, [])
        self.assertFalse(client.get("/status").json["ollamaAvailable"])


if __name__ == "__main__":
    unittest.main()
