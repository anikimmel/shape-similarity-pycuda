from shapesimilarity import procrustes_normalize_curve, procrustes_normalize_rotation
import unittest

class TestProcrustesNormalizeRotation(unittest.TestCase):
  def test_rotates_a_normalized_curve_to_match_the_rotation_of_another_normalized_curve(self):
    curve = procrustes_normalize_curve([[0, 0], [1, 0]])
    relative_curve = procrustes_normalize_curve([[0, 0], [0, 1]]).tolist()
    rotated_curve = procrustes_normalize_rotation(curve, relative_curve).tolist()
    self.assertEqual(
      list(map(lambda item: [round(item[0], 4), round(item[1], 4)], rotated_curve)),
      list(map(lambda item: [round(item[0], 4), round(item[1], 4)], rotated_curve))
    )

  def test_throws_an_error_if_the_curves_have_different_numbers_of_points(self):
    curve1 = [[0, 0], [1, 1]]
    curve2 = [[0, 0], [1, 1], [1.5], 1.5]
    with self.assertRaises(AssertionError) as context:
      procrustes_normalize_rotation(curve1, curve2)
    self.assertTrue('curve and relativeCurve must have the same length' in str(context.exception))


if __name__ == '__main__':
    unittest.main()