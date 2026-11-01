import importlib.util
import pathlib
import sys
import unittest


_MODULE_PATH = (pathlib.Path(__file__).parents[2] / "src" /
                "speech_aware_evidence_acquisition" / "core" /
                "observability_profiles.py")
_SPEC = importlib.util.spec_from_file_location("observability_profiles", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
ObservabilityProfileError = _MODULE.ObservabilityProfileError
parse_profile = _MODULE.parse_profile


class ObservabilityProfileTests(unittest.TestCase):
    def test_missing_profile_is_normal(self):
        profile = parse_profile(None)
        self.assertEqual(profile.mode, "normal")
        self.assertEqual(profile.monitor_interval_seconds, 600)
        self.assertFalse(profile.node_timing)

    def test_legacy_debug_profile_normalizes(self):
        profile = parse_profile({
            "mode": "debug",
            "client_detail": True,
            "server_metrics_interval_seconds": 600,
        })
        self.assertTrue(profile.node_timing)
        self.assertTrue(profile.trace_payloads)
        self.assertTrue(profile.flush_each_event)

    def test_typo_and_unsafe_interval_fail_closed(self):
        base = {"mode": "normal", "client_detail": False,
                "server_metrics_interval_seconds": 600}
        for bad in (
            {**base, "client_detial": False},
            {**base, "server_metrics_interval_seconds": 9},
            {**base, "mode": "debug"},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ObservabilityProfileError):
                    parse_profile(bad)


if __name__ == "__main__":
    unittest.main()
