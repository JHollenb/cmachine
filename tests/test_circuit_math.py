"""Independent NumPy/finite-difference checks of the compiled C inspection API."""
import ctypes as C
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = C.POINTER(C.c_double)


class ReadGeometry(C.Structure):
    _fields_ = [(name, C.c_double) for name in ("gate_norm", "up_norm", "dot", "eigen_plus", "eigen_minus")]


class CircuitMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        library = Path(cls.tmp.name) / "circuit.dylib"
        subprocess.run(["cc", "-shared", "-fPIC", "-std=c11", "-O1", "-ffp-contract=off",
                        str(ROOT / "src/circuit_math.c"), "-lm", "-o", str(library)], check=True)
        cls.lib = C.CDLL(str(library))
        cls.lib.cm_silu_secant.argtypes = [C.c_double, C.c_double]
        cls.lib.cm_silu_secant.restype = C.c_double
        for name in ("cm_rms_jvp", "cm_rms_vjp", "cm_rms_finite"):
            getattr(cls.lib, name).argtypes = [C.c_size_t, P, P, C.c_double, P, P] if name != "cm_rms_finite" else [C.c_size_t, P, P, P, C.c_double, P]
        cls.lib.cm_attention_finite.argtypes = [C.c_size_t, C.c_size_t, P, P, P, P, P]
        cls.lib.cm_attention_query_second.argtypes = [C.c_size_t, C.c_size_t, P, P, P, P, P, P]
        cls.lib.cm_attention_vjp.argtypes = [C.c_size_t, C.c_size_t] + [P] * 10
        cls.lib.cm_neuron_curvature.argtypes = [C.c_size_t] + [P] * 6
        cls.lib.cm_shift_commutator.argtypes = [C.c_size_t, P, P]
        cls.lib.cm_rms_second.argtypes = [C.c_size_t, P, P, C.c_double, P, P, P]
        cls.lib.cm_read_geometry.argtypes = [C.c_size_t, P, P, C.POINTER(ReadGeometry)]
        cls.lib.cm_neuron_even_odd.argtypes = [C.c_double, C.c_double, P, P]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def p(x):
        assert x.dtype == np.float64 and x.flags.c_contiguous
        return x.ctypes.data_as(P)

    def test_rms_finite_and_adjoint(self):
        rng = np.random.default_rng(431)
        for _ in range(40):
            x, y, gamma, v, w = rng.normal(size=(5, 9))
            out, jv, jt = (np.empty(9) for _ in range(3))
            self.assertEqual(self.lib.cm_rms_finite(9, self.p(x), self.p(y), self.p(gamma), 1e-6, self.p(out)), 0)
            normalize = lambda z: gamma * z / np.sqrt(np.mean(z*z) + 1e-6)
            np.testing.assert_allclose(out, normalize(y)-normalize(x), atol=3e-15, rtol=3e-14)
            self.assertEqual(self.lib.cm_rms_jvp(9, self.p(x), self.p(gamma), 1e-6, self.p(v), self.p(jv)), 0)
            self.assertEqual(self.lib.cm_rms_vjp(9, self.p(x), self.p(gamma), 1e-6, self.p(w), self.p(jt)), 0)
            self.assertAlmostEqual(jv @ w, v @ jt, places=12)
            eps = 1e-5
            np.testing.assert_allclose(jv, (normalize(x+eps*v)-normalize(x-eps*v))/(2*eps), atol=2e-9, rtol=2e-8)

    def test_prompt_free_read_geometry_and_even_odd_split(self):
        rng = np.random.default_rng(312)
        for _ in range(30):
            g, u = rng.normal(size=(2, 9))
            out = ReadGeometry()
            self.assertEqual(self.lib.cm_read_geometry(9, self.p(g), self.p(u), C.byref(out)), 0)
            eigen = np.linalg.eigvalsh((np.outer(g, u)+np.outer(u, g))/2)
            np.testing.assert_allclose([out.eigen_minus, out.eigen_plus], eigen[[0, -1]], atol=1e-14, rtol=1e-13)
            gate, up = rng.normal(size=2)
            even, odd = np.empty(1), np.empty(1)
            self.assertEqual(self.lib.cm_neuron_even_odd(gate, up, self.p(even), self.p(odd)), 0)
            np.testing.assert_allclose(even[0]+odd[0], gate/(1+np.exp(-gate))*up, atol=1e-15)
            np.testing.assert_allclose(even[0]-odd[0], (-gate)/(1+np.exp(gate))*(-up), atol=1e-15)

    def test_silu_secants_and_exact_finite_factorization(self):
        rng = np.random.default_rng(119)
        phi = lambda x: x/(1+np.exp(-x))
        for _ in range(40):
            g, u = rng.normal(size=(2, 7, 5))
            down = rng.normal(size=(5, 7))
            x0, x1 = rng.normal(size=(2, 5))
            a0, a1, b0, b1 = g @ x0, g @ x1, u @ x0, u @ x1
            secants = np.array([self.lib.cm_silu_secant(a, b) for a, b in zip(a0, a1)])
            operator = down @ ((.5*(b0+b1)*secants)[:, None]*g + (.5*(phi(a0)+phi(a1)))[:, None]*u)
            expected = down @ (phi(a1)*b1-phi(a0)*b0)
            np.testing.assert_allclose(operator @ (x1-x0), expected, atol=3e-14, rtol=3e-13)
        for x in (-1000., -2., 0., 2., 1000.):
            self.assertTrue(np.isfinite(self.lib.cm_silu_secant(x, x)))

    def test_raw_normalization_curvature_keeps_radial_terms(self):
        rng = np.random.default_rng(41)
        for _ in range(20):
            x, gamma, a, b = rng.normal(size=(4, 9))
            out = np.empty(9)
            self.assertEqual(self.lib.cm_rms_second(9, self.p(x), self.p(gamma), 1e-6, self.p(a), self.p(b), self.p(out)), 0)
            f = lambda z: gamma*z/np.sqrt(np.mean(z*z)+1e-6)
            eps = 2e-4
            expected = (f(x+eps*a+eps*b)-f(x+eps*a-eps*b)-f(x-eps*a+eps*b)+f(x-eps*a-eps*b))/(4*eps*eps)
            np.testing.assert_allclose(out, expected, atol=2e-6, rtol=3e-5)

    def test_attention_full_vjp_and_curvature(self):
        rng = np.random.default_rng(79)
        for _ in range(20):
            keys, values, dk, dv = rng.normal(size=(4, 6, 4))
            q, direction, lam = rng.normal(size=(3, 4))
            def forward(q, keys, values):
                logits = keys @ q / 2
                p = np.exp(logits-logits.max()); p /= p.sum()
                return p, p @ values
            p, a = forward(q, keys, values)
            oq, ok, ov, ol = np.empty(4), np.empty((6, 4)), np.empty((6, 4)), np.empty(6)
            self.assertEqual(self.lib.cm_attention_vjp(6, 4, *map(self.p, (p, q, keys, values, a, lam, oq, ok, ov, ol))), 0)
            eps = 1e-5
            fd = (forward(q+eps*direction, keys+eps*dk, values+eps*dv)[1]-forward(q-eps*direction, keys-eps*dk, values-eps*dv)[1])/(2*eps)
            self.assertAlmostEqual(lam @ fd, oq @ direction+np.sum(ok*dk)+np.sum(ov*dv), places=8)
            second = np.empty(4)
            self.assertEqual(self.lib.cm_attention_query_second(6, 4, *map(self.p, (p, keys, values, a, direction, second))), 0)
            eps = 2e-4
            fd2 = (forward(q+eps*direction, keys, values)[1]-2*a+forward(q-eps*direction, keys, values)[1])/eps**2
            np.testing.assert_allclose(second, fd2, atol=6e-8, rtol=2e-6)

    def test_finite_attention_extremes_and_cancellation(self):
        rng = np.random.default_rng(67)
        for offset in (-1000., 0., 1000.):
            p = rng.random(8); p /= p.sum()
            values, dv = rng.normal(size=(2, 8, 3))
            dl = rng.normal(size=8) + offset
            out = np.empty(3)
            self.assertEqual(self.lib.cm_attention_finite(8, 3, *map(self.p, (p, values, dl, dv, out))), 0)
            logits = np.log(p)+dl; changed = np.exp(logits-logits.max()); changed /= changed.sum()
            np.testing.assert_allclose(out, changed @ (values+dv)-p @ values, atol=3e-14, rtol=3e-13)

    def test_neuron_curvature(self):
        rng = np.random.default_rng(13)
        for _ in range(20):
            g, u, x, a, b = rng.normal(size=(5, 7))*.4
            out = np.empty(1)
            self.assertEqual(self.lib.cm_neuron_curvature(7, *map(self.p, (g, u, x, a, b, out))), 0)
            f = lambda z: (g @ z)/(1+np.exp(-g @ z))*(u @ z)
            eps = 1e-4
            expected = (f(x+eps*a+eps*b)-f(x+eps*a-eps*b)-f(x-eps*a+eps*b)+f(x-eps*a-eps*b))/(4*eps**2)
            np.testing.assert_allclose(out[0], expected, atol=1e-8, rtol=1e-5)

    def test_frequency_commutator_identity(self):
        rng = np.random.default_rng(5)
        for n in (3, 7, 11):
            j = rng.normal(size=(n, n))
            out = np.empty(1)
            self.assertEqual(self.lib.cm_shift_commutator(n, self.p(j), self.p(out)), 0)
            transform = np.fft.fft(np.eye(n), axis=0)/np.sqrt(n)
            fourier = transform @ j @ transform.conj().T
            z = np.exp(-2j*np.pi*np.arange(n)/n)
            expected = np.sum(np.abs(z[None, :]-z[:, None])**2*np.abs(fourier)**2)
            self.assertAlmostEqual(out[0], expected, places=10)


if __name__ == "__main__":
    unittest.main()
