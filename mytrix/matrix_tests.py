"""Module for testing general matrix class."""

import unittest

from matrix import Matrix
import exceptions as exc


class MatrixTests(unittest.TestCase):
    """Unit test functions."""

    def testInit(self):
        """Test initialiser."""
        # test initialising with valid dimensions
        m1 = Matrix(2, 2)
        self.assertTrue(m1.m == 2)
        self.assertTrue(m1.n == 2)
        self.assertTrue(m1.data == [[0, 0], [0, 0]])

        # test initialising with valid dimensions but no data attribute
        m2 = Matrix(2, 2, init=False)
        self.assertTrue(m2.m == 2)
        self.assertTrue(m2.n == 2)
        self.assertTrue(m2.data == [])

        # test initialising with invald dimensions
        with self.assertRaises(TypeError):
            Matrix(2, 'spam')
        with self.assertRaises(ValueError):
            Matrix(2, 0)

        # test initialising with invalid init
        with self.assertRaises(TypeError):
            Matrix(2, 2, 'spam')

    def testStr(self):
        """Test string method."""
        raise NotImplementedError()

    def testRepl(self):
        """Test REPL method."""
        raise NotImplementedError()

    def testAdd(self):
        """Test addition operator."""
        # test addition by matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        m3 = m1 + m2
        self.assertTrue(m3 == Matrix.fromRows([[6, 8], [10, 12]]))

        # test addition by scalar
        m4 = m1 + 1
        self.assertTrue(m4 == Matrix.fromRows([[2, 3], [4, 5]]))

        # test addition by non-conforming matrix
        m5 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            m1 + m5

        # test addition by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 + 'spam'

    def testSub(self):
        """Test subtraction operator."""
        # test subtraction by matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        m3 = m1 - m2
        self.assertTrue(m3 == Matrix.fromRows([[-4, -4], [-4, -4]]))

        # test subtraction by scalar
        m4 = m1 - 1
        self.assertTrue(m4 == Matrix.fromRows([[0, 1], [2, 3]]))

        # test subtraction by non-conforming matrix
        m5 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            m1 - m5

        # test subtraction by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 - 'spam'

    def testMul(self):
        """Test multiplication operator."""
        # test multiplication by matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        m3 = m1 * m2
        self.assertTrue(m3 == Matrix.fromRows([[19, 22], [43, 50]]))

        # test multiplication by scalar
        m4 = m1 * 2
        self.assertTrue(m4 == Matrix.fromRows([[2, 4], [6, 8]]))

        # test multiplication by non-conforming matrix
        m5 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            m1 * m5

        # test multiplication by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 * 'spam'

    def testNeg(self):
        """Test matrix negation."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = -m1
        self.assertTrue(m2 == Matrix.fromRows([[-1, -2], [-3, -4]]))

    def testEq(self):
        """Test matrix equality."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1 == Matrix.fromRows([[1, 2], [3, 4]]))
        self.assertTrue(not m1 == 'spam')
        self.assertTrue(not m1 == Matrix.fromRows([[1, 2], [3, 4], [5, 6]]))

    def testGetItem(self):
        """Test getting of matrix element."""
        # test getting element using valid key
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1[1, 1] == 4)

        # test getting element using invalid key
        with self.assertRaises(TypeError):
            m1['spam']
        # TypeError check (must me tuple) is performed before ValueError
        # check (must be length two) so m1[1] raises TypeError
        with self.assertRaises(TypeError):
            m1[1]
        with self.assertRaises(ValueError):
            m1[1, 1, 1]
        with self.assertRaises(TypeError):
            m1[1, 'spam']
        with self.assertRaises(exc.OutOfBoundsError):
            m1[-1, 1]
        with self.assertRaises(exc.OutOfBoundsError):
            m1[1, -1]
        with self.assertRaises(exc.OutOfBoundsError):
            m1[2, 1]
        with self.assertRaises(exc.OutOfBoundsError):
            m1[1, 2]

    def testSetItem(self):
        """Test setting of matrix element."""
        # test setting element using valid key
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m1[1, 1] = 5
        self.assertTrue(m1 == Matrix.fromRows([[1, 2], [3, 5]]))

        # test setting element using invalid key
        with self.assertRaises(TypeError):
            m1['spam'] = 5
        # TypeError check (must me tuple) is performed before ValueError
        # check (must be length two) so m1[1] raises TypeError
        with self.assertRaises(TypeError):
            m1[1] = 5
        with self.assertRaises(ValueError):
            m1[1, 1, 1] = 5
        with self.assertRaises(TypeError):
            m1[1, 'spam'] = 5
        with self.assertRaises(exc.OutOfBoundsError):
            m1[-1, 1] = 5
        with self.assertRaises(exc.OutOfBoundsError):
            m1[1, -1] = 5
        with self.assertRaises(exc.OutOfBoundsError):
            m1[2, 1] = 5
        with self.assertRaises(exc.OutOfBoundsError):
            m1[1, 2] = 5

    def testSubset(self):
        """Test matrix subsetting."""
        # test subsetting matrix using valid rows/cols
        m1 = Matrix.fromRows([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        m2 = m1.subset([0, 2], [1])
        self.assertTrue(m2 == Matrix.fromRows([[2], [8]]))

        # test subsetting matrix using invalid rows/cols
        with self.assertRaises(TypeError):
            m1.subset([0, 2], 'spam')
        with self.assertRaises(ValueError):
            m1.subset([0, 2], [])
        with self.assertRaises(exc.OutOfBoundsError):
            m1.subset([-1, 2], [1])
        with self.assertRaises(exc.OutOfBoundsError):
            m1.subset([0, 2], [-1])
        with self.assertRaises(exc.OutOfBoundsError):
            m1.subset([0, 3], [1])
        with self.assertRaises(exc.OutOfBoundsError):
            m1.subset([0, 2], [3])

    def testTranspose(self):
        """Test matrix transposition."""
        # test transposition
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.transpose() == Matrix.fromRows([[1, 3], [2, 4]]))

        # test involution property of transposition
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.transpose().transpose() == m1)


if __name__ == "__main__":
    unittest.main()