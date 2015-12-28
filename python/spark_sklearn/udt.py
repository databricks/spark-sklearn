
import numpy as np
from scipy.sparse import csr_matrix

from pyspark.sql.types import ByteType, IntegerType, ArrayType, DoubleType, \
    StructField, StructType, UserDefinedType

class CSRVectorUDT(UserDefinedType):
    """
    SQL user-defined type (UDT) for scipy.sparse.csr_matrix (vectors only, not matrices).

     .. note:: Experimental
    """

    @classmethod
    def sqlType(cls):
        return StructType([
            StructField("type", ByteType(), False),
            StructField("size", IntegerType(), True),
            StructField("indices", ArrayType(IntegerType(), False), True),
            StructField("values", ArrayType(DoubleType(), False), True)])

    @classmethod
    def module(cls):
        return "spark_sklearn"

    def serialize(self, obj):
        if isinstance(obj, csr_matrix):
            assert obj.shape[0] == 1, \
                "cannot serialize csr_matrix which has shape[1]=%d" % obj.shape[1]
            assert len(obj.indptr) == 2, \
                "cannot serialize csr_matrix which has len(indptr)!=2"
            size = obj.shape[1]
            indices = [int(i) for i in obj.indices]
            values = [float(v) for v in obj.data]
            return (0, size, indices, values)
        else:
            raise TypeError("cannot serialize %r of type %r" % (obj, type(obj)))

    def deserialize(self, datum):
        assert len(datum) == 4, \
            "CSRVectorUDT.deserialize given row with length %d but requires 4" % len(datum)
        tpe = datum[0]
        assert tpe==0, "CSRVectorUDT.deserialize requires sparse format"
        data = np.array(datum[3])
        indices = np.array(datum[2])
        size = datum[1]
        indptr = np.array([0, len(indices)])
        return csr_matrix((data, indices, indptr), shape=(1, size))

    def simpleString(self):
        return "csrvec"
