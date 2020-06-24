"""Module for general matrix class."""

from math import log10
import functools
import operator

from copy import deepcopy
from random import randrange

import mytrix.exceptions as exc
import mytrix.vector as vec


class Matrix():
    """A class to represent a general matrix."""

    str_decimal_places = 3
    str_bools_as_int = False

    def __init__(self, m, n, data):
        """Initalise matrix dimensions and contents."""
        self.m = m
        self.n = n
        self.data = data

    def __repr__(self):
        """Generate reproducible representation of matrix."""
        class_name = self.__class__.__name__
        data = ',\r\n'.join(
            ['    [' + ', '.join(list(map(str, row))) + ']'
             for row in self.data])
        return f'{class_name}({self.m}, {self.n}, [\r\n{data}\r\n])'

    def __iter__(self):
        """Iterate over elements of the matrix row-wise."""
        for i in range(self.m):
            for j in range(self.n):
                yield self[i, j]

    def __copy__(self):
        """Create a shallow copy of this matrix.

        Creates a new instance of Matrix but with data referencing the data
        of the original matrix.
        """
        return self.__class__(self.m, self.n, self.data)

    def __deepcopy__(self, memodict=None):
        """Create a deep copy of this matrix.

        Creates a new instance of Matrix with data copied from the original
        matrix.
        """
        return self.__class__(self.m, self.n, deepcopy(self.data))

    def __iadd__(self, obj):
        """Add a matrix to this matrix, modifying it in the process."""
        # calls __add__
        tmp = self + obj
        self.data = tmp.data
        return self

    def __isub__(self, obj):
        """Subtract a matrix from this matrix, modifying it in the process."""
        # calls __sub__
        tmp = self - obj
        self.data = tmp.data
        return self

    def __imul__(self, obj):
        """Right multiply this matrix by another and modify.

        Multiply this matrix by another matrix (on the right), modifying
        it in the process.
        """
        # calls __mul__
        tmp = self * obj
        self.data = tmp.data
        self.m, self.n = tmp.dim
        return self

    def __ifloordiv__(self, obj):
        """Divide this matrix by a scalar, modifying it in the process."""
        # calls __floordiv__
        tmp = self // obj
        self.data = tmp.data
        return self

    def __itruediv__(self, obj):
        """Divide this matrix by a scalar, modifying it in the process."""
        # calls __truediv__
        tmp = self / obj
        self.data = tmp.data
        return self

    def __radd__(self, obj):
        """Implement reflected addition."""
        # calls __add__
        return self + obj

    def __rsub__(self, obj):
        """Implement reflected subtraction."""
        # calls __sub__
        return -self + obj

    def __rmul__(self, obj):
        """Implement reflected multiplication."""
        # calls __mul__
        if isinstance(obj, vec.Vector):
            TypeError(
                "Vector-matrix multiplication cannot be performed"
            )
        return self * obj

    @property
    def dim(self):
        """Get matrix dimensions as tuple."""
        return self.m, self.n

    def __getitem__(self, key):
        """Get element in (i, j)th position."""
        self.__check_key_validity(key)
        return self.data[key[0]][key[1]]

    def __setitem__(self, key, val):
        """Set element in (i, j)th position."""
        self.__check_key_validity(key)
        self.data[key[0]][key[1]] = val

    def __check_key_validity(self, key):
        """Validate keys for __getitem__() and __setitem__() methods."""
        if not isinstance(key, tuple):
            raise TypeError("key must be a tuple")
        if len(key) != 2:
            raise ValueError("key must be of length two")
        if not (isinstance(key[0], int) and isinstance(key[1], int)):
            raise TypeError("elements of key must be integers")
        if not ((0 <= key[0] < self.m) and (0 <= key[1] < self.n)):
            raise exc.OutOfBoundsError("key is out of bounds")

    def subset(self, rows, cols):
        """Extract subset of data and columns and form into a new matrix."""
        # validation on data/cols
        if not (isinstance(rows, list) and isinstance(cols, list)):
            raise TypeError("arguments must be lists")
        if len(rows) == 0 or len(cols) == 0:
            raise ValueError("subset cannot be empty")
        # validation on elements of data/cols
        for i, elem in enumerate(rows + cols):
            if not isinstance(elem, int):
                raise TypeError("elements of data/cols must be integers")
            # if element represents a row
            if i < len(rows):
                if not 0 <= elem < self.m:
                    raise exc.OutOfBoundsError("key is out of bounds")
            else:
                if not 0 <= elem < self.n:
                    raise exc.OutOfBoundsError("key is out of bounds")
        # subset matrix
        data = [[self[r, c] for c in cols] for r in rows]
        return self.__class__(len(data), len(cols), data)

    def transpose(self):
        """Transpose this matrix and return the result."""
        data = [list(col) for col in zip(*self.data)]
        return self.__class__(self.n, self.m, data)

    def is_symmetric(self):
        """Return True if and only if this matrix is symmetric."""
        return self.all_equal(self.transpose())

    def is_skew_symmetric(self):
        """Return True if and only if this matrix is skew-symmetric."""
        return self.all_equal(-self.transpose())

    def toeplitz_decomposition(self):
        """Apply the Toeplitz decomposition to this matrix.

        Decompose this matrix into the sum of a symmetric and skew-symmetric
        matrix, returning the result as a tuple.
        """
        if self.m != self.n:
            raise exc.DecompositionError("non-square matrices do not have a " +
                                         "a Toeplitz decomposition")
        # TODO: test for decomposition directly using the parity of elements
        try:
            divisor = 2 if isinstance(self, IntegerMatrix) else 2.
            sym = (self + self.transpose()) / divisor
            skew = (self - self.transpose()) / divisor
        except ValueError:
            raise exc.DecompositionError("Toeplitz decomposition does not " +
                                         "exist over the integers")
        return sym, skew

    def qr_decomposition(self):
        """Apply the QR decomposition to this matrix.

        Decompose this matrix into the product of an orthogonal and
        upper triangular matrix, returning the result as a tuple.
        """
        if self.m != self.n:
            raise NotImplementedError('QR decomposition not yet available ' +
                                      'for non-square matrices')
        orig_basis = [vec.Vector.fromMatrixColumn(self, j)
                      for j in range(self.m)]
        orthog_basis, norm_basis = [], []
        for j in range(self.m):
            u = orig_basis[j]
            for k in range(j):
                u -= orig_basis[j].project_onto(orthog_basis[k])
            orthog_basis.append(u)
            norm_basis.append(u.normalise())
        Q = Matrix.fromVectors(norm_basis)
        R = Q.transpose() * self
        return Q, R

    def row_reduce(self):
        """Return the row-reduced form of this matrix."""
        res = self.row_echelon()
        for i in range(1, res.m):
            for j in range(res.n):
                if res[i, j] == 1:
                    for k in range(i):
                        constant = res[k, j]
                        res.data[k] = [elem_k - elem_i * constant
                                       for elem_i, elem_k in
                                       zip(res.data[i], res.data[k])]
                    break
        return res

    def invert(self):
        """Calculate the inverse of a non-singular matrix.

        This method currently implements Gaussian elimination.
        """
        if self.m != self.n:
            raise exc.LinearAlgebraError("cannot invert a non-square matrix")
        if self.determinant == 0:
            raise exc.LinearAlgebraError("cannot invert a singular matrix")
        # TODO: implement block matrices in their own method
        block_rows = [r1 + r2 for r1, r2 in
                      zip(self.data, self.makeIdentity(self.m).data)]
        inverse_block = Matrix.fromRows(block_rows).row_reduce()
        return inverse_block.subset([i for i in range(self.m)],
                                    [j + self.n for j in range(self.n)])

    @property
    def inverse(self):
        """Calculate the inverse of an invertible matrix as a property."""
        return self.invert()

    @classmethod
    def makeRandom(cls, m, n, min=0, max=1):
        """Create random matrix.

        Make a random matrix of dimension m by n with elements chosen
        independently and uniformly from the interval (min, max).
        """
        Matrix.validate_dimensions(m, n)
        data = [[randrange(min, max) for j in range(n)] for i in range(m)]
        return RealMatrix(m, n, data)

    @staticmethod
    def fromRows(data):
        """Make a matrix from a list of data."""
        m = len(data)
        n = len(data[0])
        # check that data structure is valid
        if any([len(row) != n for row in data[1:]]):
            raise ValueError("inconsistent row lengths")
        # check that data types are inconsistent
        t = type(data[0][0])
        if any(any(type(e) is not t for e in row[(i == 0):])
               for i, row in enumerate(data)):
            raise TypeError("inconsistent element types")
        # dispatch to childern based on type
        if t is bool:
            return BooleanMatrix(m, n, data)
        elif t is int:
            return IntegerMatrix(m, n, data)
        if t is float:
            return RealMatrix(m, n, data)

    @classmethod
    def fromCols(cls, data):
        """Make a matrix from a list of data."""
        m = len(data[0])
        # check that list of data is valid
        if any([len(col) != m for col in data[1:]]):
            raise ValueError("inconsistent column lengths")
        return Matrix.fromRows(data).transpose()

    @classmethod
    def fromList(cls, elems, **kwargs):
        """Make matrix from list.

        Make a matrix from a list of elements, filling along data,
        when given at least one dimension of the matrix.
        """
        if not ('m' in kwargs or 'n' in kwargs):
            raise ValueError("at least one of m and n must be specified")
        m = kwargs.get('m')
        n = kwargs.get('n')
        num_elems = len(elems)
        if m is None:
            m = num_elems // n
        elif n is None:
            n = num_elems // m
        elif m * n != num_elems:
            raise ValueError("dimension does not match number of elements in"
                             "list")

        data = [elems[i * n: i * (n + 1)] for i in range(m)]
        return Matrix(m, n, data)

    @classmethod
    def fromVectors(cls, vectors):
        """Make matrix from a list of vectors."""
        data = [[v[i] for i in range(v.m)] for v in vectors]
        return Matrix.fromCols(data)

    @classmethod
    def set_str_precision(cls, dp=3):
        """Set/reset the decimal precision used for the __str__() magic."""
        cls.str_decimal_places = dp

    @staticmethod
    def is_numeric(obj):
        """Check if a given object is of a numeric type.

        Note that since bool inherits from int, that this will accept
        Boolean values
        """
        return isinstance(obj, (int, float, complex))

    @staticmethod
    def validate_dimensions(m, n):
        """Check whether a pair of matrix dimensions are valid."""
        if not (isinstance(m, int) and isinstance(n, int)):
            raise TypeError("dimensions must be integral")
        if m <= 0 or n <= 0:
            raise ValueError("dimensions must be positive")


class BooleanMatrix(Matrix):
    """A class to represent a Boolean matrix."""

    def __str__(self):
        """Generate text representation of matrix."""
        if self.str_bools_as_int:
            s = '\n'.join([' '.join([
                f"{elem:>{5}}" for elem in row])
                          for row in self.data])
        else:
            s = '\n'.join([' '.join([
                f"{int(elem)}" for elem in row])
                          for row in self.data])
        return s + '\n'

    def _boolean_operation(self, obj, op):
        """Perform a generic Boolean operation."""
        if isinstance(obj, Matrix):
            if self.m != obj.m or self.n != obj.n:
                raise exc.ComformabilityError(
                        "matrices must have the same dimensions")
            if not isinstance(obj, BooleanMatrix):
                raise TypeError("operation only exists for Boolean matrices")
            data = [[op(self[i, j], obj[i, j])
                    for j in range(self.n)]
                    for i in range(self.m)]
        elif isinstance(obj, bool):
            data = [[op(self[i, j], obj)
                    for j in range(self.n)]
                    for i in range(self.m)]
        else:
            raise TypeError(
                    "operation can't be performed with object of type " +
                    type(obj).__name__)
        return self.__class__(self.m, self.n, data)

    def __and__(self, obj):
        """Compute Boolean AND of this matrix and another valid object."""
        return self._boolean_operation(obj, operator.__and__)

    def __or__(self, obj):
        """Compute Boolean OR of this matrix and another valid object."""
        return self._boolean_operation(obj, operator.__or__)

    def __xor__(self, obj):
        """Compute Boolean XOR of this matrix and another valid object."""
        return self._boolean_operation(obj, operator.__xor__)

    def __add__(self, obj):
        """Add a valid object to this matrix using Boolean XOR."""
        return self ^ obj

    def __sub__(self, obj):
        """Subtract a valid object from this matrix using Boolean XOR."""
        return self ^ obj

    def __mul__(self, obj):
        """Multiply this matrix by a valid object using Boolean AND."""
        return self & obj

    @property
    def determinant(self):
        """Calculate the determinant of a square matrix.

        This method currently implements the Laplace expansion
        """
        if self.m != self.n:
            raise exc.LinearAlgebraError("cannot calculate the determinant of"
                                         "a non-square matrix")
        if self.m == 1:
            return self[0, 0]
        # TODO: can we choose a better row/column to improve efficiency
        return functools.reduce(
            lambda x, y: x ^ y,
            [self[0, j] and
                self.subset([i for i in range(1, self.m)],
                            [k for k in range(self.n) if k != j]).determinant
             for j in range(self.n)],
        )

    @staticmethod
    def makeZero(m, n):
        """Make a zero matrix of dimension m by n."""
        Matrix.validate_dimensions(m, n)
        data = [[False for j in range(n)] for i in range(m)]
        return BooleanMatrix(m, n, data)

    @staticmethod
    def makeIdentity(m):
        """Make an identity matrix of dimension m by m."""
        Matrix.validate_dimensions(m, m)
        data = [[i == j for j in range(m)] for i in range(m)]
        return BooleanMatrix(m, m, data)


class NumericMatrix(Matrix):
    """A class to represent a numeric matrix."""

    def __eq__(self, obj):
        """Elementwise equality."""
        if isinstance(obj, Matrix):
            if self.m != obj.m or self.n != obj.n:
                raise exc.ComformabilityError(
                        "matrices must have the same dimensions")
            if type(self) is not type(obj):
                raise TypeError("matrices must be the same type")
            data = [[self[i, j] == obj[i, j]
                    for j in range(self.n)]
                    for i in range(self.m)]
        elif Matrix.is_numeric(obj):
            self._validate_scalar(obj)
            data = [[self[i, j] == obj
                    for j in range(self.n)]
                    for i in range(self.m)]
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj).__name__ +
                    " to matrix")
        return BooleanMatrix(self.m, self.n, data)

    def __neq__(self, obj):
        """Elementwise equality."""
        if isinstance(obj, Matrix):
            if self.m != obj.m or self.n != obj.n:
                raise exc.ComformabilityError(
                        "matrices must have the same dimensions")
            if type(self) is not type(obj):
                raise TypeError("matrices must be the same type")
            data = [[self[i, j] != obj[i, j]
                    for j in range(self.n)]
                    for i in range(self.m)]
        elif Matrix.is_numeric(obj):
            self._validate_scalar(obj)
            data = [[self[i, j] != obj
                    for j in range(self.n)]
                    for i in range(self.m)]
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj).__name__ +
                    " to matrix")
        return BooleanMatrix(self.m, self.n, data)

    def __add__(self, obj):
        """Add a valid object to this matrix and return the result.

        Doesn't modify the current matrix. Valid objects include other matrices
        and numeric scalars
        """
        if isinstance(obj, Matrix):
            if self.m != obj.m or self.n != obj.n:
                raise exc.ComformabilityError(
                        "matrices must have the same dimensions")
            if type(self) is not type(obj):
                raise TypeError("matrices must be the same type")
            data = [[self[i, j] + obj[i, j]
                    for j in range(self.n)]
                    for i in range(self.m)]
        elif Matrix.is_numeric(obj):
            self._validate_scalar(obj)
            data = [[self[i, j] + obj
                    for j in range(self.n)]
                    for i in range(self.m)]
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj).__name__ +
                    " to matrix")
        return self.__class__(self.m, self.n, data)

    def __sub__(self, obj):
        """Subtract a valid object from this matrix and return the result.

        Doesn't modify the current matrix. Valid objects include other matrices
        and numeric scalars
        """
        if isinstance(obj, Matrix):
            if self.m != obj.m or self.n != obj.n:
                raise exc.ComformabilityError(
                        "matrices must have the same dimensions")
            if type(self) is not type(obj):
                raise TypeError(
                        "matrices must be the same type")
            data = [[self[i, j] - obj[i, j]
                    for j in range(self.n)]
                    for i in range(self.m)]
        elif Matrix.is_numeric(obj):
            self._validate_scalar(obj)
            data = [[self[i, j] - obj
                    for j in range(self.n)]
                    for i in range(self.m)]
        else:
            raise TypeError(
                    "cannot subtract object of type " + type(obj).__name__ +
                    " to matrix")
        return self.__class__(self.m, self.n, data)

    def __mul__(self, obj):
        """Multiply this matrix by a valid object and return the result.

        Doesn't modify the current matrix. Valid objects include other matrices
        vectors, and numeric scalars. In the case where the other object is a
        matrix, multiplication occurs with the current matrix on the left-hand
        side.
        """
        if isinstance(obj, Matrix):
            if self.n != obj.m:
                raise exc.ComformabilityError(
                        "inner matrix dimensions must match")
            if type(self) is not type(obj):
                raise TypeError(
                        "matrices must be the same type")
            data = [[sum([self[i, k] * obj[k, j] for k in range(self.n)])
                    for j in range(obj.n)]
                    for i in range(self.m)]
            return self.__class__(self.m, obj.n, data)
        elif isinstance(obj, vec.Vector):
            raise NotImplementedError("vector type system not implemented")
            if self.n != obj.m:
                raise exc.ComformabilityError(
                        "number of matrix columns much match vector length")
            data = [sum([self[i, k] * obj[k] for k in range(self.n)])
                    for i in range(self.m)]
            return vec.Vector(self.m, data)
        elif Matrix.is_numeric(obj):
            self._validate_scalar(obj)
            data = [[self[i, j] * obj
                    for j in range(self.n)]
                    for i in range(self.m)]
            return self.__class__(self.m, self.n, data)
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj).__name__ +
                    " to matrix")

    def __pos__(self):
        """Unary positive. Included for symmetry only."""
        data = [[+self[i, j] for j in range(self.n)] for i in range(self.m)]
        return self.__class__(self.m, self.n, data)

    def __neg__(self):
        """Negate all elements of the matrix."""
        data = [[-self[i, j] for j in range(self.n)] for i in range(self.m)]
        return self.__class__(self.m, self.n, data)

    def __floordiv__(self, obj):
        """Divide this matrix by a scalar.

        Doesn't modify the current matrix
        """
        if Matrix.is_numeric(obj):
            data = [[self[i, j] // obj
                    for j in range(self.n)]
                    for i in range(self.m)]
            return IntegerMatrix(self.m, self.n, data)
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj).__name__ +
                    " to matrix")

    def all_equal(self, mtrx):
        """Evaluate whether two matrices are equal."""
        if not isinstance(mtrx, Matrix):
            return False
        if not (self.m == mtrx.m and self.n == mtrx.n):
            return False
        if not type(self) == type(mtrx):
            return False
        for i in range(self.m):
            for j in range(self.n):
                if self[i, j] != mtrx[i, j]:
                    return False
        return True

    @property
    def determinant(self):
        """Calculate the determinant of a square matrix.

        This method currently implements the Laplace expansion
        """
        if self.m != self.n:
            raise exc.LinearAlgebraError("cannot calculate the determinant of"
                                         "a non-square matrix")
        if self.m == 1:
            return self[0, 0]
        # TODO: can we choose a better row/column to improve efficiency
        return sum([self[0, j] * (-1 if j % 2 else 1) *
                    self.subset([i for i in range(1, self.m)],
                    [k for k in range(self.n) if k != j]).determinant
                   for j in range(self.n)])

    @staticmethod
    def hadamard(A, B):
        """Calculate the Hadamard product of two matrices."""
        if not all(isinstance(M, IntegerMatrix) for M in (A, B)):
            raise TypeError("can only Hadamard two matrices")
        if type(A) is not type(B):
            raise TypeError(
                    "matrices must be the same type")
        if A.m != B.m or A.n != B.n:
            raise exc.ComformabilityError(
                "matrices must have the same dimensions")
        data = [[A[i, j] * B[i, j]
                 for j in range(A.n)]
                for i in range(A.m)]
        return A.__class__(A.m, B.m, data)


class IntegerMatrix(NumericMatrix):
    """A class to represent a integer matrix."""

    def __str__(self):
        """Generate text representation of matrix."""
        largest_element = max(self)
        length = int(log10(largest_element)) + 1

        s = '\n'.join([' '.join([
            f"{elem:>{length}}" for elem in row])
                      for row in self.data])
        return s + '\n'

    def __truediv__(self, obj):
        """Divide this matrix by a scalar.

        Doesn't modify the current matrix
        """
        if not isinstance(obj, int):
            raise TypeError(
                    "cannot divide by object of type " + type(obj).__name__ +
                    " to matrix")
        if any(self % obj != 0):
            raise ValueError(
                    "not all matrix elements are divisible by " + str(obj))
        data = [[self[i, j] // obj
                for j in range(self.n)]
                for i in range(self.m)]
        return IntegerMatrix(self.m, self.n, data)

    def __mod__(self, obj):
        """Compute the modulus of this matrix when divided by a scalar.

        Doesn't modify the current matrix
        """
        if not isinstance(obj, int):
            raise TypeError(
                    "cannot modulo object of type " + type(obj).__name__ +
                    " to matrix")
        data = [[self[i, j] % obj
                for j in range(self.n)]
                for i in range(self.m)]
        return IntegerMatrix(self.m, self.n, data)

    def row_echelon(self):
        """Return the row-echelon form of this matrix."""
        # TODO: This can be refactored for better efficiency
        if all([all([self[i, j] == 0 for j in range(self.n)])
                for i in range(self.m)]):
            return Matrix.makeZero(self.m, self.n)
        res = deepcopy(self)
        i, j = 0, 0
        while i < res.m and j < res.n:
            # Use R2 to make pivot non-zero
            if res[i, j] == 0:
                found_non_zero = False
                for k in range(i, res.m):
                    if res[k, j] != 0:
                        found_non_zero = True
                        break
                if not found_non_zero:
                    j += 1
                    continue
                res.data[i], res.data[k] = res.data[k], res.data[i]
            # Use R3 to make pivot one
            if res[i, j] != 1:
                if any([elem % res[i, j] != 0 for elem in res.data[i]]):
                    raise ValueError
                res.data[i] = [elem / res[i, j] for elem in res.data[i]]
            # Use R1 to eliminate entries below the pivot
            for k in range(i + 1, res.m):
                if res[k, j] != 0:
                    constant = res[k, j] / res[i, j]
                    res.data[k] = [elem_k - elem_i * constant
                                   for elem_i, elem_k in
                                   zip(res.data[i], res.data[k])]
            i, j = i + 1, j + 1
        return res

    @staticmethod
    def _validate_scalar(obj):
        """Check whether a scalar is a member of the correct field."""
        if not isinstance(obj, int):
            raise TypeError("scalar must be an integer")

    @staticmethod
    def makeZero(m, n):
        """Make a zero matrix of dimension m by n."""
        Matrix.validate_dimensions(m, n)
        data = [[0 for j in range(n)] for i in range(m)]
        return IntegerMatrix(m, n, data)

    @staticmethod
    def makeIdentity(m):
        """Make an identity matrix of dimension m by m."""
        Matrix.validate_dimensions(m, m)
        data = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
        return IntegerMatrix(m, m, data)


class RealMatrix(NumericMatrix):
    """A class to represent a real matrix."""

    def __str__(self):
        """Generate text representation of matrix."""
        largest_element = max(self)
        integer_part_length = int(log10(largest_element)) + 1
        length = integer_part_length + self.str_decimal_places + 1

        s = '\n'.join([' '.join([
            f"{elem:{length}.{self.str_decimal_places}f}"
            for elem in row])
                      for row in self.data])
        return s + '\n'

    def all_near(self, mtrx, tol=10e-8):
        """
        Evaluate whether two matrices have all entries approximately equal.

        Check whether corresponding elements of two matrices are within
        a specified tolerance of one another.
        """
        if not isinstance(mtrx, Matrix):
            return False
        if not (self.m == mtrx.m and self.n == mtrx.n):
            return False
        for i in range(self.m):
            for j in range(self.n):
                if abs(self[i, j] - mtrx[i, j]) > tol:
                    return False
        return True

    def __truediv__(self, obj):
        """Divide this matrix by a scalar.

        Doesn't modify the current matrix
        """
        if not isinstance(obj, float):
            data = [[self[i, j] / obj
                    for j in range(self.n)]
                    for i in range(self.m)]
            return self.__class__(self.m, self.n, data)
        else:
            raise TypeError(
                    "cannot add object of type " + type(obj).__name__ +
                    " to matrix")

    def row_echelon(self):
        """Return the row-echelon form of this matrix."""
        # TODO: This can be refactored for better efficiency
        if all([all([self[i, j] == 0 for j in range(self.n)])
                for i in range(self.m)]):
            return Matrix.makeZero(self.m, self.n)
        res = deepcopy(self)
        i, j = 0, 0
        while i < res.m and j < res.n:
            # Use R2 to make pivot non-zero
            if res[i, j] == 0:
                found_non_zero = False
                for k in range(i, res.m):
                    if res[k, j] != 0:
                        found_non_zero = True
                        break
                if not found_non_zero:
                    j += 1
                    continue
                res.data[i], res.data[k] = res.data[k], res.data[i]
            # Use R3 to make pivot one
            if res[i, j] != 1:
                res.data[i] = [elem / res[i, j] for elem in res.data[i]]
            # Use R1 to eliminate entries below the pivot
            for k in range(i + 1, res.m):
                if res[k, j] != 0:
                    constant = res[k, j] / res[i, j]
                    res.data[k] = [elem_k - elem_i * constant
                                   for elem_i, elem_k in
                                   zip(res.data[i], res.data[k])]
            i, j = i + 1, j + 1
        return res

    @staticmethod
    def _validate_scalar(obj):
        """Check whether a scalar is a member of the correct field."""
        if not isinstance(obj, float):
            raise TypeError("scalar must be real")

    @staticmethod
    def makeZero(m, n):
        """Make a zero matrix of dimension m by n."""
        Matrix.validate_dimensions(m, n)
        data = [[0. for j in range(n)] for i in range(m)]
        return RealMatrix(m, n, data)

    @staticmethod
    def makeIdentity(m):
        """Make an identity matrix of dimension m by m."""
        Matrix.validate_dimensions(m, m)
        data = [[1. if i == j else 0. for j in range(m)] for i in range(m)]
        return RealMatrix(m, m, data)
