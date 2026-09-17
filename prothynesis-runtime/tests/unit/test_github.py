import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from prothynesis_runtime.github.oauth import (
    CredentialStore,
    GitHubAuthResult,
    GitHubOAuthClient,
)
from prothynesis_runtime.github.mock import MockGitHubClient


def test_github_oauth_client_authorization_url():
    client = GitHubOAuthClient("client_id", "client_secret", "http://localhost/callback")
    url = client.get_authorization_url(state="my_state")
    assert url.startswith(client.AUTHORIZATION_URL)
    assert "client_id=client_id" in url
    assert "redirect_uri=http" in url
    assert "state=my_state" in url


def test_mock_github_client_authorization():
    client = MockGitHubClient()
    url = client.authorize(state="my_state")
    assert "client_id=mock" in url
    assert "state=my_state" in url


@patch("prothynesis_runtime.github.oauth.requests.post")
def test_credential_store_save_load_cycle(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "token123",
        "token_type": "bearer",
        "scope": "read:user",
        "expires_in": 3600,
    }
    mock_post.return_value = mock_response
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = os.path.join(tmpdir, "creds.json")
        client = GitHubOAuthClient("client_id", "client_secret", "http://localhost/callback")
        result = client.exchange_code("code123")
        assert result is not None
        store = CredentialStore(storage_path)
        saved = store.save_credentials(result)
        assert saved is True
        loaded = store.load_credentials()
        assert loaded is not None
        assert loaded.access_token == "token123"


def test_credential_store_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = os.path.join(tmpdir, "creds.json")
        store = CredentialStore(storage_path)
        auth_result = GitHubAuthResult(
            access_token="token123",
            token_type="bearer",
            scope="read:user",
            github_id=42,
            username="testuser",
            expires_at=None,
        )
        store.save_credentials(auth_result)
        assert store.load_credentials() is not None
        cleared = store.clear_credentials()
        assert cleared is True
        assert store.load_credentials() is None
