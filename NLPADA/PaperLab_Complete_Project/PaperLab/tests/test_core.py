import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from paperlab.tokenizer import ByteBPE, PAD
from paperlab.model import MiniGPT, Config, cross_entropy
from paperlab.training import load_corpus, make_batch, save_checkpoint, load_checkpoint
from paperlab.retrieval import Passage, BM25, split_page, read_papers
from paperlab.ranking import RetrievalIndex
from paperlab.rag import Candidate, locate_span, answer, FlanGenerator
from paperlab.generators import ExtractiveGenerator, OllamaGenerator
from paperlab.service import STARTER, PaperService


class TokenizerTests(unittest.TestCase):
    def test_unicode_roundtrip(self):
        t = ByteBPE.train(["research research paper paper", "hello world"], 280)
        for text in ["", "नमस्ते भारत 🙂", "a\n\tb  c", "x² + β ≤ 43.5", "café\x00"]:
            self.assertEqual(t.decode(t.encode(text, boundaries=True)), text)

    def test_reproducible_merges_and_serialization(self):
        t = ByteBPE.train(["abab abc abc", "abab"], 275)
        self.assertEqual(t.merges, ByteBPE.train(["abab", "abab abc abc"], 275).merges)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"bpe.json"; t.save(p)
            self.assertEqual(ByteBPE.load(p).encode("abc 😀"), t.encode("abc 😀"))

    def test_invalid_merge(self):
        with self.assertRaises(ValueError): ByteBPE([(400, 0)])


class ModelTests(unittest.TestCase):
    def test_manual_gradients_finite_difference(self):
        model = MiniGPT(Config(11, 4, 8, 2, 2), dtype=np.float64)
        x = np.array([[1, 2, 1], [3, 1, 4]])
        y = np.array([[2, 1, 5], [1, 4, -1]])
        logits, cache = model.forward(x, with_cache=True)
        _, derivative = cross_entropy(logits, y, gradient=True)
        grads = model.backward(derivative, cache)
        rng = np.random.default_rng(5)
        errors = []
        for key, weights in model.weights.items():
            for flat in rng.choice(weights.size, min(4, weights.size), replace=False):
                idx = np.unravel_index(flat, weights.shape)
                value = weights[idx]; eps = 1e-5
                weights[idx] = value + eps
                plus = cross_entropy(model.forward(x), y)
                weights[idx] = value - eps
                minus = cross_entropy(model.forward(x), y)
                weights[idx] = value
                numerical = (plus - minus) / (2 * eps)
                self.assertAlmostEqual(numerical, grads[key][idx], delta=2e-5, msg=f"Gradient {key}{idx}")
                errors.append(abs(numerical - grads[key][idx]))
        self.assertEqual(set(grads), set(model.weights))
        print(f"\nFinite differences: {len(errors)} coordinates, max absolute error {max(errors):.3g}")

    def test_attention_cannot_see_future(self):
        model = MiniGPT(Config(12, 4, 8, 2, 2))
        a = model.forward(np.array([[1, 2, 3, 4]]))
        b = model.forward(np.array([[1, 2, 9, 10]]))
        np.testing.assert_allclose(a[:, :2], b[:, :2], atol=1e-7)

    def test_padding_has_no_loss_gradient(self):
        logits = np.zeros((1, 3, 6))
        loss, grad = cross_entropy(logits, np.array([[2, 4, -1]]), gradient=True)
        self.assertAlmostEqual(loss, np.log(6))
        np.testing.assert_array_equal(grad[:, 2], 0)

    def test_bad_token_ids_rejected(self):
        m = MiniGPT(Config(5, 3, 4, 1, 1))
        with self.assertRaises(ValueError): m.forward(np.array([[-1]]))
        with self.assertRaises(ValueError): m.forward(np.array([[1.5]]))

    def test_checkpoint_preserves_logits_and_tokenizer(self):
        tok = ByteBPE.train(["hello hello"], 266)
        m = MiniGPT(Config(tok.vocab_size, 8, 8, 2, 1))
        x = np.array([tok.encode("hello")])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"model.npz";save_checkpoint(p, m, tok, {"seed": 17})
            restored, rt, report = load_checkpoint(p)
            np.testing.assert_array_equal(m.forward(x), restored.forward(x))
            self.assertEqual(rt.merges, tok.merges)
            self.assertEqual(report["seed"], 17)


class DataTests(unittest.TestCase):
    def test_duplicate_cross_split_rejected(self):
        rows = [{"id":"a","split":"train","text":"Same text"},
                {"id":"b","split":"validation","text":" same  TEXT "}]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"corpus.jsonl";p.write_text("\n".join(map(json.dumps,rows)))
            with self.assertRaises(ValueError):load_corpus(p)

    def test_batches_do_not_cross_documents(self):
        x, y = make_batch([(np.array([1,2,3]),0),(np.array([4,5]),0)],4)
        np.testing.assert_array_equal(x, [[1,2,PAD,PAD],[4,PAD,PAD,PAD]])
        np.testing.assert_array_equal(y, [[2,3,-1,-1],[5,-1,-1,-1]])

    def test_chunk_overlap_and_page_identity(self):
        self.assertEqual(list(split_page("a b c d e f",4,2)),["a b c d","c d e f"])
        with self.assertRaises(ValueError):list(split_page("x",4,4))
        passages, issues = read_papers(STARTER/"papers")
        self.assertFalse(issues)
        self.assertEqual(len({(p.file,p.page) for p in passages}),6)


class RetrievalAndEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.p = [Passage("a.pdf",1,1,"The alarm threshold was 80 centimetres."),
                  Passage("a.pdf",1,2,"The alarm threshold was 80 centimetres."),
                  Passage("b.pdf",2,1,"Median filtering reduces a distance spike.")]

    def test_all_ranking_modes_and_distinct_pages(self):
        for mode in ("bm25","tfidf","hybrid"):
            index = RetrievalIndex(self.p, mode=mode)
            self.assertEqual(index.search("alarm threshold",1)[0].passage.file,"a.pdf")
            hits=index.search("alarm median",3,distinct_pages=True)
            self.assertEqual(len(hits),2)
            self.assertFalse(index.search("quasar uranium",3))

    def test_numeric_span_boundaries(self):
        for guess, context in [("43","143"),("43","-43"),("0.5","0.50"),("50","50%"),("43","43.8"),("yes","yes")]:
            self.assertIsNone(locate_span(guess,context))
        self.assertEqual(locate_span("80 centimetres","It was 80 centimetres.").group(),"80 centimetres")

    def test_hallucinated_or_invented_context_rejected(self):
        class Fake:
            def propose(self,q,p):return Candidate("999 metres",p)
        index=BM25(self.p)
        self.assertNotEqual(answer(index,Fake(),"alarm threshold").status,"answered")
        class Invented:
            def propose(self,q,p):return Candidate("80 centimetres","Invented context: 80 centimetres")
        self.assertNotEqual(answer(index,Invented(),"alarm threshold").status,"answered")

    def test_low_coverage_and_attribute_abstention(self):
        index=BM25(self.p)
        self.assertEqual(answer(index,ExtractiveGenerator(),"uranium submarine").status,"low_retrieval")
        self.assertNotEqual(answer(index,ExtractiveGenerator(),"Who manufactured the distance sensor?").status,"answered")

    def test_flan_budget_keeps_question(self):
        class Tokenizer:
            def encode(self,text,**kwargs):return text.split()
        g=FlanGenerator();g.tokenizer=Tokenizer()
        visible=g._fit("What threshold?"," ".join(["word"]*100),budget=60)
        self.assertTrue(visible)
        self.assertLessEqual(len(g.tokenizer.encode(g._prompt("What threshold?",visible))),60)
        with self.assertRaises(ValueError):g._fit("word "*100,"x",budget=60)

    def test_ollama_is_local(self):
        with self.assertRaises(ValueError):OllamaGenerator(url="https://example.com")


class ServiceTests(unittest.TestCase):
    def test_upload_roundtrip_duplicates_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=PaperService(tmp);content=(STARTER/"papers/sensor_filter.pdf").read_bytes()
            with self.assertRaises(ValueError):s.upload("../escape.pdf",content)
            with self.assertRaises(ValueError):s.upload("bad.pdf",b"not a pdf")
            self.assertEqual(s.upload("new.pdf",content)["status"],"added")
            self.assertEqual(s.upload("new.pdf",content)["status"],"already_present")
            with self.assertRaises(FileExistsError):s.upload("new.pdf",(STARTER/"papers/window_overlap.pdf").read_bytes())
            self.assertEqual(len(PaperService(tmp).status()["documents"]),4)


if __name__ == "__main__": unittest.main()
