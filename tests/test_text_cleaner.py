import unittest

from app.utils.text_cleaner import clean_text


class TextCleanerTestCase(unittest.TestCase):
    def test_none_or_blank_text_returns_empty_string(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(" \n\t "), "")

    def test_removes_control_characters_and_zero_width_chars(self):
        raw_text = "ab\u0001cd\u0002ef\u200bgh\ufeffij"
        self.assertEqual(clean_text(raw_text), "abcdefghij")

    def test_normalizes_line_endings_and_whitespace(self):
        raw_text = "line1\r\nline2\rline3\t\tvalue\u00a0\u3000end"
        self.assertEqual(clean_text(raw_text), "line1\nline2\nline3 value end")

    def test_collapses_multiple_spaces_and_extra_blank_lines(self):
        raw_text = "a   b\n\n\n\nc    d"
        self.assertEqual(clean_text(raw_text), "a b\n\nc d")

    def test_keeps_normal_characters(self):
        raw_text = "中文 ABC 123，保留正常标点。"
        self.assertEqual(clean_text(raw_text), raw_text)


if __name__ == "__main__":
    unittest.main()
