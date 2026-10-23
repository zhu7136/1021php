"""CPU-only GitHub Release URL and redirect validation; no external calls."""
from pathlib import Path
import sys
import unittest
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import git_smoke_v8 as smoke

RELEASE = 'https://github.com'+smoke.GITHUB_RELEASE_PATH


class GitHubReleaseSourceTests(unittest.TestCase):
    def test_exact_public_release_allowed(self):
        smoke.check_url(RELEASE)

    def test_release_asset_cdn_allowed(self):
        smoke.check_url('https://release-assets.githubusercontent.com/github-production-release-asset/test?signature=fixture')

    def test_legacy_asset_cdn_allowed(self):
        smoke.check_url('https://objects.githubusercontent.com/github-production-release-asset/test?signature=fixture')

    def test_other_repository_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(RELEASE.replace('zhu7136/1021php', 'other/repo'))

    def test_other_release_tag_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(RELEASE.replace('hud04-data-v8-56-clips', 'unapproved-tag'))

    def test_other_asset_name_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(RELEASE.replace(smoke.ARCHIVE_NAME, 'other.zip'))

    def test_plain_http_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(RELEASE.replace('https:', 'http:'))

    def test_deceptive_cdn_suffix_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url('https://release-assets.githubusercontent.com.evil.example/test')

    def test_deceptive_github_suffix_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(RELEASE.replace('github.com', 'github.com.evil.example'))

    def test_embedded_credentials_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(RELEASE.replace('https://', 'https://user:password@'))

    def test_unapproved_port_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(RELEASE.replace('github.com/', 'github.com:8443/'))

    def test_fragment_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(RELEASE+'#fragment')

    def test_github_query_rejected(self):
        with self.assertRaises(ValueError):
            smoke.check_url(RELEASE+'?unexpected=fixture')

    def test_redirect_to_unapproved_domain_rejected(self):
        req = urllib.request.Request(RELEASE)
        with self.assertRaises(ValueError):
            smoke.SafeRedirect().redirect_request(req,None,302,'Found',{},'https://example.com/file.zip')

    def test_redirect_to_release_cdn_allowed_without_auth(self):
        req = urllib.request.Request(RELEASE)
        redirected = smoke.SafeRedirect().redirect_request(req,None,302,'Found',{},
            'https://release-assets.githubusercontent.com/github-production-release-asset/test?signature=fixture')
        self.assertIsNotNone(redirected)
        self.assertIsNone(redirected.get_header('Authorization'))


if __name__ == '__main__':
    unittest.main()