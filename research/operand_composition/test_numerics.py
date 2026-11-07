"""Model-free checks for the independent attention/RoPE audit equations."""
import numpy as np
from numerics import full_read, rope


def test_zero_query_reads_the_mean_with_grouped_values():
    keys=np.arange(3*2*128,dtype=float).reshape(3,2,128)
    values=np.zeros_like(keys)
    values[:,0,:]=np.asarray([1.,4.,7.])[:,None]
    values[:,1,:]=np.asarray([3.,9.,15.])[:,None]
    result=full_read(np.zeros(1536),keys,values).reshape(12,128)
    np.testing.assert_allclose(result[:6],4.)
    np.testing.assert_allclose(result[6:],9.)


def test_half_split_rope_preserves_each_heads_norm_and_is_clocked():
    value=np.random.default_rng(2107).normal(size=(12,128))
    np.testing.assert_array_equal(rope(value,0,1e6),value.reshape(-1))
    rotated=rope(value,137,1e6).reshape(12,128)
    np.testing.assert_allclose(np.linalg.norm(rotated,axis=1),np.linalg.norm(value,axis=1),rtol=1e-14)
    assert not np.array_equal(rotated,value)
