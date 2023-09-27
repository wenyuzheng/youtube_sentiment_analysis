import unittest

import processors as p

class TestCase(unittest.TestCase):
  def test_normalize_spaces(self):
    input = " a \r b "
    expected = "a b"
    self.assertEqual(p.normalize_spaces(input), expected)

  def test_remove_irrelevant_char(self):
    input = "aña and bełla Mc'donalds +- &*(%$#@!()/\\"
    expected = "aña and be la Mc'donalds           !"
    self.assertEqual(p.remove_irrelevant_char(input), expected)

  def test_remove_new_lines(self):
    input = "a \n b"
    expected = "a   b"
    self.assertEqual(p.remove_new_lines(input), expected)

  def test_normalize_repeating_characters(self):
    input = "YESSS ARMMYYYY sooooo gooooooood"
    expected = "YESS ARMMYY soo good"
    self.assertEqual(p.normalize_repeating_characters(input), expected)

  def test_remove_urls(self):
    input = "this has url: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    expected = "this has url: "
    self.assertEqual(p.remove_urls(input), expected)

    input = "this has url: http://www.youtube.com/watch?v=dQw4w9WgXcQ"
    expected = "this has url: "
    self.assertEqual(p.remove_urls(input), expected)

    input = "this has url: www.youtube.com/watch?v=dQw4w9WgXcQ"
    expected = "this has url: "
    self.assertEqual(p.remove_urls(input), expected)

if __name__ == '__main__':
    unittest.main()