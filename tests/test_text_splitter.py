import unittest

from app.utils.text_splitter import split_text


class TextSplitterTestCase(unittest.TestCase):
    def test_empty_text_returns_empty_list(self):
        self.assertEqual(split_text(""), [])
        self.assertEqual(split_text("   "), [])
        self.assertEqual(split_text(None), [])

    def test_single_chunk_when_text_is_shorter_than_chunk_size(self):
        self.assertEqual(split_text("hello", chunk_size=10, overlap=2), ["hello"])

    def test_split_text_with_overlap(self):
        result = split_text("abcdefghij", chunk_size=4, overlap=1)
        self.assertEqual(result, ["abcd", "defg", "ghij"])

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with self.assertRaises(ValueError):
            split_text("hello", chunk_size=4, overlap=4)

    def test_invalid_chunk_size_or_overlap_raises_error(self):
        with self.assertRaises(ValueError):
            split_text("hello", chunk_size=0, overlap=0)

        with self.assertRaises(ValueError):
            split_text("hello", chunk_size=4, overlap=-1)

    def test_splitter_does_not_dead_loop(self):
        result = split_text("abcdefghij", chunk_size=4, overlap=3)
        self.assertEqual(result, ["abcd", "bcde", "cdef", "defg", "efgh", "fghi", "ghij"])


if __name__ == "__main__":
    unittest.main()
