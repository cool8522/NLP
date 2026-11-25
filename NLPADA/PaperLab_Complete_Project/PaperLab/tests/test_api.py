import tempfile
import unittest
from fastapi.testclient import TestClient
from paperlab.api import create_app
from paperlab.service import STARTER


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.client=TestClient(create_app(self.temp.name))
        self.headers={"X-PaperLab":"1"}
    def tearDown(self):
        self.client.close();self.temp.cleanup()
    def test_dashboard_and_paper(self):
        self.assertEqual(self.client.get("/").status_code,200)
        self.assertEqual(self.client.get("/api/status").json()["searchable_pages"],6)
        self.assertTrue(self.client.get("/api/papers/sensor_filter.pdf").content.startswith(b"%PDF"))
        self.assertEqual(self.client.get("/api/papers/missing.pdf").status_code,404)
    def test_answer_and_empty_input(self):
        r=self.client.post("/api/ask",headers=self.headers,json={"question":"What was the alarm threshold?"})
        self.assertEqual(r.status_code,200);self.assertIn("80 centimetres",r.json()["text"])
        self.assertEqual(self.client.post("/api/ask",headers=self.headers,json={"question":" "}).status_code,400)
        self.assertEqual(self.client.post("/api/ask",headers=self.headers,json={"question":"x","k":0}).status_code,422)
    def test_mutation_origin_guard(self):
        self.assertEqual(self.client.post("/api/ask",json={"question":"x"}).status_code,403)
        self.assertEqual(self.client.post("/api/ask",headers={**self.headers,"Origin":"https://other.example"},json={"question":"x"}).status_code,403)
    def test_upload_endpoint(self):
        r=self.client.post("/api/upload",headers={**self.headers,"X-Filename":"added.pdf"},content=(STARTER/"papers/sensor_filter.pdf").read_bytes())
        self.assertEqual(r.status_code,200)
        self.assertEqual(len(self.client.get("/api/status").json()["documents"]),4)


if __name__ == "__main__":unittest.main()
