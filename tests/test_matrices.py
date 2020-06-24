"""Module for testing general matrix class."""

from math import sqrt

from copy import copy, deepcopy
import unittest

import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mytrix.matrix import Matrix, IntegerMatrix, NumericMatrix  # noqa
from mytrix import Vector  # noqa
import mytrix.exceptions as exc  # noqa


class MatrixTests(unittest.TestCase):
    """Unit test functions."""

    def testCopy(self):
        """Test shallow and deep copying."""
        # test shallow copying
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = copy(m1)
        self.assertTrue(m2 is not m1)
        m2[1, 1] = 5
        self.assertTrue(m1[1, 1] == 5)

        # test deep copying
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = deepcopy(m1)
        self.assertTrue(m2 is not m1)
        m2[1, 1] = 5
        self.assertTrue(m1[1, 1] == 4)

    def testStr(self):
        """Test string method."""
        m1 = Matrix.fromRows([[1, 20], [300, 4000]])
        self.assertTrue(str(m1) == '   1   20\n' +
                                   ' 300 4000\n')

        m2 = Matrix.fromRows([[1., 20.], [300., 4000.]])
        self.assertTrue(str(m2) == '   1.000   20.000\n' +
                                   ' 300.000 4000.000\n')

        # test decimal precision
        Matrix.set_str_precision(2)
        self.assertTrue(str(m2) == '   1.00   20.00\n' +
                                   ' 300.00 4000.00\n')

    def testRepr(self):
        """Test repr method."""
        m1 = Matrix.fromRows([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(repr(m1) == "IntegerMatrix(2, 3, [\r\n" +
                                    "    [1, 2, 3],\r\n" +
                                    "    [4, 5, 6]\r\n" +
                                    "])")
        self.assertTrue(eval(repr(m1)).all_equal(m1))

    def testIter(self):
        """Test iteration method."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        for i, e in enumerate(m1):
            self.assertTrue(e == i + 1)

    def testAllEqual(self):
        """Test eq method."""
        # test equality
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.all_equal(m1))

        # test non-equality
        m2 = Matrix.fromRows(([1, 2], [3, 5]))
        m3 = Matrix.fromRows(([1, 2, 2], [3, 4, 4]))
        self.assertFalse(m1.all_equal('spam'))
        self.assertFalse(m1.all_equal(m2))
        self.assertFalse(m1.all_equal(m3))

    def testAllNear(self):
        """Test approximate equality."""
        # test approximate equality
        m1 = Matrix.fromRows([[1., 2.], [3., 4.]])
        m2 = Matrix.fromRows([[1., 2.], [3., 4. + 10e-10]])
        self.assertTrue(m1.all_near(m2))

        # test approximate in-equality
        m3 = Matrix.fromRows([[1., 2.], [3., 4. + 10e-6]])
        self.assertFalse(m1.all_near(m3))

        # test custom tolerance
        self.assertTrue(m1.all_near(m3, tol=10e-4))

        # test non-quality
        m4 = Matrix.fromRows(([1., 2., 2.], [3., 4., 4.]))
        self.assertFalse(m1.all_near('spam'))
        self.assertFalse(m1.all_near(m4))

    def testAdd(self):
        """Test addition operator."""
        # test addition by matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        m3 = m1 + m2
        self.assertTrue(m3.all_equal(Matrix.fromRows([[6, 8], [10, 12]])))

        # test addition by scalar
        m4 = m1 + 1
        self.assertTrue(m4.all_equal(Matrix.fromRows([[2, 3], [4, 5]])))

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
        self.assertTrue(m3.all_equal(Matrix.fromRows([[-4, -4], [-4, -4]])))

        # test subtraction by scalar
        m4 = m1 - 1
        self.assertTrue(m4.all_equal(Matrix.fromRows([[0, 1], [2, 3]])))

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
        self.assertTrue(m3.all_equal(Matrix.fromRows([[19, 22], [43, 50]])))

        # test multiplication by non-square (but conforming) matrix
        m3 = Matrix.fromRows([[5, 6, 7], [8, 9, 10]])
        m4 = m1 * m3
        self.assertTrue(m4.all_equal(Matrix.fromRows([[21, 24, 27], [47, 54, 61]])))

        # test multiplication by vector
        v1 = Vector.fromList([1, 2])
        self.assertTrue(m1 * v1 == Vector.fromList([5, 11]))

        # test multiplication by scalar
        m5 = m1 * 2
        self.assertTrue(m5.all_equal(Matrix.fromRows([[2, 4], [6, 8]])))

        # test multiplication by non-conforming matrix
        m6 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            m1 * m6

        # test multiplication by non-conforming vector
        v2 = Vector.fromList([1, 2, 3])
        with self.assertRaises(exc.ComformabilityError):
            m1 * v2

        # test multiplication by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 * 'spam'

    def testDiv(self):
        """Test division operator."""
        # test true division by matrix
        m1 = Matrix.fromRows([[1., 2.], [3., 4.]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        with self.assertRaises(TypeError):
            m1 / m2

        # test true division by scalar
        m3 = m1 / 2
        self.assertTrue(m3.all_equal(Matrix.fromRows([[.5, 1.], [1.5, 2.]])))

        # test true division by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 / 'spam'

        # test floor division by matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        with self.assertRaises(TypeError):
            m1 // m2

        # test floor division by scalar
        m3 = m1 // 2
        self.assertTrue(m3.all_equal(Matrix.fromRows([[0, 1], [1, 2]])))

        # test floor division by non-matrix/numeric object
        with self.assertRaises(TypeError):
            m1 // 'spam'

    def testArithmeticAssignment(self):
        """Test matrix arithmetic using assignment magics."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])

        # test addition
        m1 += m2
        self.assertTrue(m1.all_equal(Matrix.fromRows([[6, 8], [10, 12]])))
        m1 += 1
        self.assertTrue(m1.all_equal(Matrix.fromRows([[7, 9], [11, 13]])))

        # test subtraction
        m1 = Matrix.fromRows([[1, 2], [3, 4]])  # reset m1
        m1 -= m2
        self.assertTrue(m1.all_equal(Matrix.fromRows([[-4, -4], [-4, -4]])))
        m1 -= 1
        self.assertTrue(m1.all_equal(Matrix.fromRows([[-5, -5], [-5, -5]])))

        # test multiplication
        m1 = Matrix.fromRows([[1, 2], [3, 4]])  # reset m1
        m1 *= m2
        self.assertTrue(m1.all_equal(Matrix.fromRows([[19, 22], [43, 50]])))
        m1 *= 2
        self.assertTrue(m1.all_equal(Matrix.fromRows([[38, 44], [86, 100]])))

        # test division
        m1 = Matrix.fromRows([[1., 2.], [3., 4.]])  # reset m1
        # inplace floor division seems like a bad idea as it changes type
        # m1 //= 2
        # self.assertTrue(m1.all_equal(Matrix.fromRows([[0, 1], [1, 2]])))
        m1 /= 2
        self.assertTrue(m1.all_equal(Matrix.fromRows([[0.5, 1.], [1.5, 2.]])))

    def testArithmeticReflection(self):
        """Test matrix arithmetic using reflection magics."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])

        # test addition
        m2 = 1 + m1
        self.assertTrue(m2.all_equal(Matrix.fromRows([[2, 3], [4, 5]])))

        # test subtraction
        m2 = 1 - m1
        self.assertTrue(m2.all_equal(Matrix.fromRows([[0, -1], [-2, -3]])))

        # test multiplication
        m2 = 2 * m1
        self.assertTrue(m2.all_equal(Matrix.fromRows([[2, 4], [6, 8]])))

    def testPos(self):
        """Test unary positive method."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.all_equal(+m1))

    def testNeg(self):
        """Test matrix negation."""
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = -m1
        self.assertTrue(m2.all_equal(Matrix.fromRows([[-1, -2], [-3, -4]])))

    def testDim(self):
        """Test matrix dimensions."""
        m1 = Matrix.fromRows([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(m1.dim == (2, 3))

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
        self.assertTrue(m1.all_equal(Matrix.fromRows([[1, 2], [3, 5]])))

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
        self.assertTrue(m2.all_equal(Matrix.fromRows([[2], [8]])))

        # test subsetting matrix using invalid rows/cols
        with self.assertRaises(TypeError):
            m1.subset([0, 2], 'spam')
        with self.assertRaises(ValueError):
            m1.subset([0, 2], [])
        with self.assertRaises(TypeError):
            m1.subset([0, .5], [1])
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
        self.assertTrue(m1.transpose().all_equal(Matrix.fromRows([[1, 3], [2, 4]])))

        # test involution property of transposition
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.transpose().transpose().all_equal(m1))

    def testSymmetry(self):
        """Test matrix symmetry."""
        # test symmetry
        m1 = Matrix.fromRows([[1, 2], [2, 4]])
        self.assertTrue(m1.is_symmetric())
        m2 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(not m2.is_symmetric())

        # test skew-symmetry
        m3 = Matrix.fromRows([[0, 2], [-2, 0]])
        self.assertTrue(m3.is_skew_symmetric())
        self.assertTrue(not m2.is_skew_symmetric())

    def testToeplitzDecomposition(self):
        """Test Toeplitz decomposition."""
        # test decomposition on square matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        sym, skew = m1.toeplitz_decomposition()
        self.assertTrue(sym.is_symmetric())
        self.assertTrue(skew.is_skew_symmetric())

        # test decomposition on non-square matrix
        m2 = Matrix.fromRows([[1, 2], [3, 4], [5, 6]])
        with self.assertRaises(exc.DecompositionError):
            m2.toeplitz_decomposition()

    def testQRDecomposition(self):
        """Test QR decomposition."""
        # test decomposition on square matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        Q, R = m1.qr_decomposition()
        self.assertTrue(Q.all_near(Matrix.fromRows([
            [1 / sqrt(10), 3 / sqrt(10)],
            [3 / sqrt(10), -1 / sqrt(10)]
        ])))
        self.assertTrue(m1.all_near(Q * R))

        # test decomposition on non-square matrix
        m2 = Matrix.fromRows([[1, 2]])
        with self.assertRaises(NotImplementedError):
            m2.qr_decomposition()

    def testRowReduction(self):
        """Test reduction to row-reduced and row-echelon form."""
        # test reduction to row-echelon form
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.row_echelon().all_equal(Matrix.fromRows([[1, 2], [0, 1]])))

        # test reduction on reflection matrix
        m2 = Matrix.fromRows([[0, 1], [1, 0]])
        self.assertTrue(m2.row_echelon().all_equal(IntegerMatrix.makeIdentity(2)))

        # test reduction on matrix with zero row
        m3 = Matrix.fromRows([[0, 0], [1, 0]])
        self.assertTrue(m3.row_echelon().all_equal(Matrix.fromRows([[1, 0], [0, 0]])))

        # test reduction to row-echelon form on the zero matrix
        m4 = IntegerMatrix.makeZero(2, 2)
        self.assertTrue(m4.row_echelon().all_equal(IntegerMatrix.makeZero(2, 2)))

        # test reduction to row-echelon form on the matrix with only one row
        m5 = Matrix.fromRows([[1, 2, 3, 4]])
        self.assertTrue(m5.row_reduce().all_equal(Matrix.fromRows([[1, 2, 3, 4]])))

        # test reduction to row-echelon form on the matrix with only one column
        m6 = Matrix.fromRows([[1], [2], [3], [4]])
        self.assertTrue(m6.row_reduce() == Matrix.fromRows([[1], [0],
                                                            [0], [0]]))

        # test idempotency of reduction to row-echelon form
        self.assertTrue(m1.row_echelon().all_equal(m1.row_echelon().row_echelon()))

        # test row reduction
        self.assertTrue(m1.row_reduce().all_equal(IntegerMatrix.makeIdentity(2)))

        # test row reduction on the zero matrix
        self.assertTrue(m4.row_reduce().all_equal(IntegerMatrix.makeZero(2, 2)))

        # test row reduction on the matrix with only one row
        m3 = Matrix.fromRows([[1, 2, 3, 4]])
        self.assertTrue(m5.row_reduce().all_equal(Matrix.fromRows([[1, 2, 3, 4]])))

        # test row reduction on the matrix with only one column
        self.assertTrue(m6.row_reduce() == Matrix.fromRows([[1], [0],
                                                            [0], [0]]))

        # test idempotency of reduction to row-echelon form
        self.assertTrue(m1.row_reduce().all_equal(m1.row_reduce().row_reduce()))

    def testDeterminant(self):
        """Test calculation of determinant for square matrices."""
        # test determinant on square matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.determinant == -2)

        # test determinant on identity matrix
        m2 = IntegerMatrix.makeIdentity(2)
        self.assertTrue(m2.determinant == 1)

        # test determinant on non-square matrix
        m3 = Matrix.fromRows([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(exc.LinearAlgebraError):
            m3.determinant()

        # test determinant on a singular square matrix
        m1 = Matrix.fromRows([[1, 2], [2, 4]])
        self.assertTrue(m1.determinant == 0)

    def testInversion(self):
        """Test inversion of non-singular matrices."""
        # test inversion of a non-singular matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        self.assertTrue(m1.invert().all_equal(Matrix.fromRows([[-4, 2], [3, -1]]) / 2))

        # test inversion of the identity matrix
        m2 = IntegerMatrix.makeIdentity(2)
        self.assertTrue(m2.determinant.all_equal(1))

        # test inversion of a non-square matrix
        m3 = Matrix.fromRows([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(exc.LinearAlgebraError):
            m3.invert()

        # test inversion of a singular matrix
        m4 = Matrix.fromRows([[1, 2], [2, 4]])
        with self.assertRaises(exc.LinearAlgebraError):
            m4.invert()

        # test inversion using the property method
        self.assertTrue(m1.inverse.all_equal(Matrix.fromRows([[-4, 2], [3, -1]]) / 2))

    def testHadamard(self):
        """Test Hadamard product of matrices."""
        # test Hadamard with matrix
        m1 = Matrix.fromRows([[1, 2], [3, 4]])
        m2 = Matrix.fromRows([[5, 6], [7, 8]])
        m3 = NumericMatrix.hadamard(m1, m2)
        self.assertTrue(m3.all_equal(Matrix.fromRows([[5, 12], [21, 32]])))

        # test Hadamard with non-conforming matrix
        m4 = Matrix.fromRows([[9, 10]])
        with self.assertRaises(exc.ComformabilityError):
            NumericMatrix.hadamard(m1, m4)

        # test Hadamard with non-matrix/numeric object
        with self.assertRaises(TypeError):
            NumericMatrix.hadamard(m1, 'spam')
