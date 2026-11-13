"""Independent numeric references for compiled C expansion operators."""
import ctypes as C
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = C.POINTER(C.c_double)


class Complex(C.Structure):
    _fields_ = [("re", C.c_double), ("im", C.c_double)]


class PolyAudit(C.Structure):
    _fields_ = [(name, C.c_double) for name in ("max_spectral_error", "phase_average_loss_l2", "spatial_energy")]


class Expansions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / "expansions.so"
        subprocess.run(["cc", "-shared", "-fPIC", "-std=c11", "-O1", "-ffp-contract=off",
                        str(ROOT/"src/circuit_math.c"), str(ROOT/"src/circuit_expansions.c"), "-lm", "-o", str(library)], check=True)
        cls.lib = C.CDLL(str(library))
        for name in ("cm_gelu", "cm_gelu_first", "cm_gelu_second"):
            f = getattr(cls.lib, name); f.argtypes = [C.c_double]; f.restype = C.c_double
        cls.lib.cm_gelu_secant.argtypes = [C.c_double, C.c_double]; cls.lib.cm_gelu_secant.restype = C.c_double
        cls.lib.cm_ssm_kernel.argtypes = [C.c_size_t, C.c_size_t] + [P]*5
        cls.lib.cm_ssm_cut.argtypes = [C.c_size_t]*3 + [P]*6
        cls.lib.cm_ssm_path.argtypes = [C.c_size_t]*2 + [P]*3 + [C.c_size_t]*2 + [P]
        cls.lib.cm_ssm_finite.argtypes = [C.c_size_t] + [P]*9
        cls.lib.cm_consumer_metric.argtypes = [C.c_size_t]*2 + [P]*5
        cls.lib.cm_unpatchify.argtypes = [C.c_size_t]*2 + [P]*4
        cls.lib.cm_polyphase_audit.argtypes = [C.c_size_t]*2 + [P]*3 + [C.POINTER(PolyAudit)]
        cls.lib.cm_polyphase_combine.argtypes = [C.c_size_t]*2 + [C.POINTER(Complex)]*2
        cls.lib.cm_dft2.argtypes = [C.c_size_t]*2 + [C.POINTER(Complex)]*2 + [C.c_int]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def p(x):
        assert x.dtype == np.float64 and x.flags.c_contiguous
        return x.ctypes.data_as(P)

    @staticmethod
    def z(x):
        assert x.dtype == np.complex128 and x.flags.c_contiguous
        return x.ctypes.data_as(C.POINTER(Complex))

    def test_gelu_derivatives_secant_and_symmetry(self):
        for x in np.linspace(-5, 5, 39):
            x = float(x); eps = 1e-4
            f, fp, fpp = self.lib.cm_gelu, self.lib.cm_gelu_first, self.lib.cm_gelu_second
            self.assertAlmostEqual(f(x)-f(-x), x, places=13)
            np.testing.assert_allclose(fp(x), (f(x+eps)-f(x-eps))/(2*eps), atol=2e-9, rtol=2e-7)
            np.testing.assert_allclose(fpp(x), (fp(x+eps)-fp(x-eps))/(2*eps), atol=7e-9, rtol=2e-7)
            self.assertAlmostEqual(self.lib.cm_gelu_secant(x, x+1e-12), fp(x), places=10)
        for x in (-1000., 1000.):
            self.assertTrue(np.isfinite(self.lib.cm_gelu(x)))
            self.assertEqual(self.lib.cm_gelu_second(x), 0)

    def test_scan_kernel_factorization_and_zero_retention(self):
        rng = np.random.default_rng(713)
        for t, n, cut in [(11, 3, 5), (9, 2, 2), (7, 1, 4)]:
            r = rng.random((t, n)); r[cut, 0] = 0
            b, c = rng.normal(size=(2, t, n)); dt = rng.random(t); u = rng.normal(size=t)
            kernel = np.empty((t, t)); left = np.empty((t-cut, n)); entry = np.empty((n, cut))
            self.assertEqual(self.lib.cm_ssm_kernel(t, n, *map(self.p, (r, b, c, dt, kernel))), 0)
            self.assertEqual(self.lib.cm_ssm_cut(t, n, cut, *map(self.p, (r, b, c, dt, left, entry))), 0)
            state = np.zeros(n); reads = []
            for i in range(t):
                state = r[i]*state + dt[i]*b[i]*u[i]; reads.append(c[i] @ state)
            np.testing.assert_allclose(kernel @ u, reads, atol=2e-15, rtol=2e-14)
            np.testing.assert_allclose(left @ entry, kernel[cut:, :cut], atol=2e-15, rtol=2e-14)
            self.assertLessEqual(np.linalg.matrix_rank(kernel[cut:, :cut], tol=1e-12), n)
            paths = np.empty(n); injection = np.ascontiguousarray(dt[:, None]*b*u[:, None])
            self.assertEqual(self.lib.cm_ssm_path(t, n, self.p(r), self.p(injection), self.p(c), 1, t-1, self.p(paths)), 0)
            np.testing.assert_allclose(paths, injection[1]*np.prod(r[2:], axis=0)*c[-1], atol=1e-15)

    def test_finite_scan_terms_retain_all_producer_changes(self):
        rng = np.random.default_rng(29); n = 9
        r0, r1 = rng.random((2, n)); h0, h1, i0, i1 = rng.normal(size=(4, n))
        y0 = np.ascontiguousarray((r0*h0+i0).astype(np.float32).astype(np.float64))
        y1 = np.ascontiguousarray((r1*h1+i1).astype(np.float32).astype(np.float64))
        out = np.empty((4, n))
        self.assertEqual(self.lib.cm_ssm_finite(n, *map(self.p, (r0, r1, h0, h1, i0, i1, y0, y1, out))), 0)
        np.testing.assert_allclose(out.sum(axis=0), y1-y0, atol=5e-16)
        self.assertGreater(np.max(np.abs(out[3])), 0)

    def test_consumer_metric_is_not_activation_energy(self):
        rng = np.random.default_rng(198); j = rng.normal(size=(4, 12)); e = rng.normal(size=12); w = rng.random(4)
        signed = np.empty(4); energy = np.empty(1)
        self.assertEqual(self.lib.cm_consumer_metric(4, 12, *map(self.p, (j, e, w, signed, energy))), 0)
        np.testing.assert_allclose(signed, j @ e, atol=2e-15)
        self.assertAlmostEqual(energy[0], e @ (j.T @ np.diag(w) @ j) @ e, places=12)
        w[0] = -1
        self.assertEqual(self.lib.cm_consumer_metric(4, 12, *map(self.p, (j, e, w, signed, energy))), -1)

    def test_polyphase_matches_numpy_spatial_dft_and_keeps_statistics(self):
        rng = np.random.default_rng(57)
        for h, w in [(1, 1), (3, 5), (4, 2)]:
            phase = rng.normal(size=(4, h, w)); scale = rng.random(4)+.1; bias = rng.normal(size=4)
            spatial = np.empty((2*h, 2*w)); audit = PolyAudit()
            self.assertEqual(self.lib.cm_unpatchify(h, w, *map(self.p, (phase, scale, bias, spatial))), 0)
            expected = np.empty_like(spatial)
            transformed = phase*scale[:, None, None]+bias[:, None, None]
            for a in range(2):
                for b in range(2):
                    expected[a::2, b::2] = transformed[2*a+b]
            np.testing.assert_array_equal(spatial, expected)
            spectra = np.ascontiguousarray(np.fft.fft2(transformed, axes=(-2, -1)))
            combined = np.empty((2*h, 2*w), dtype=np.complex128)
            self.assertEqual(self.lib.cm_polyphase_combine(h, w, self.z(spectra), self.z(combined)), 0)
            np.testing.assert_allclose(combined, np.fft.fft2(expected), atol=3e-14, rtol=3e-13)
            self.assertEqual(self.lib.cm_polyphase_audit(h, w, self.p(phase), self.p(scale), self.p(bias), C.byref(audit)), 0)
            self.assertLess(audit.max_spectral_error, 1e-12)
            self.assertAlmostEqual(audit.phase_average_loss_l2, np.linalg.norm(transformed-transformed.mean(axis=0)), places=12)

    def test_two_dimensional_transform_inverse(self):
        rng = np.random.default_rng(62)
        x = np.ascontiguousarray(rng.normal(size=(3, 7))+1j*rng.normal(size=(3, 7)))
        spectrum, back = np.empty_like(x), np.empty_like(x)
        self.assertEqual(self.lib.cm_dft2(3, 7, self.z(x), self.z(spectrum), 0), 0)
        self.assertEqual(self.lib.cm_dft2(3, 7, self.z(spectrum), self.z(back), 1), 0)
        np.testing.assert_allclose(back, x, atol=3e-15, rtol=3e-14)


if __name__ == "__main__":
    unittest.main()
