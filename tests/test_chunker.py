import unittest

from vectordb.services import chunk_text


class ChunkerTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(chunk_text("   "), [])

    def test_short_text_unchanged(self):
        self.assertEqual(chunk_text("hello world", 10, 2), ["hello world"])

    def test_overlap_and_coverage(self):
        text = " ".join(str(i) for i in range(100))
        chunks = chunk_text(text, 40, 10)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c.split()) <= 40 for c in chunks))
        self.assertEqual(chunks[0].split()[-10:], chunks[1].split()[:10])
        self.assertEqual(chunks[-1].split()[-1], "99")

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", 5, 5)


if __name__ == "__main__":
    unittest.main()
