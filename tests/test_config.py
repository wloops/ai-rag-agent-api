import unittest

from app.core.config import Settings


class SettingsTestCase(unittest.TestCase):
    def test_resolved_cors_allow_origins_splits_comma_separated_values(self):
        settings = Settings(
            database_url="sqlite:///test.db",
            secret_key="test-secret",
            cors_allow_origins="https://rag.restflux.online, https://preview.restflux.online ",
        )

        self.assertEqual(
            settings.resolved_cors_allow_origins,
            ["https://rag.restflux.online", "https://preview.restflux.online"],
        )

    def test_resolved_cors_allow_origins_falls_back_to_default_domain(self):
        settings = Settings(
            database_url="sqlite:///test.db",
            secret_key="test-secret",
            cors_allow_origins="   ",
        )

        self.assertEqual(
            settings.resolved_cors_allow_origins,
            ["https://rag.restflux.online"],
        )
