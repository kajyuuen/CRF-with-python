import sys
import unittest
sys.path.append('..')
import load

class LoadTest(unittest.TestCase):
    def test_load_test(self):
        f = open("test.txt")
        wd, pd, ctd, etd = load.load_data(f)
        f.close()

        self.assertEqual(wd, [["Peter", "Blackburn"], ["1966", "World", "Cup"]])
        self.assertEqual(pd, [["NNP", "NNP"], ["CD", "NNP", "NNP"]])
        self.assertEqual(ctd, [["I-NP", "I-NP"], ["I-NP", "I-NP", "I-NP"]])
        #self.assertEqual(etd, [["I-PER", "I-PER"], ["I-MISC", "I-MISC", "I-MISC"]])


if __name__ == '__main__':
    unittest.main()
