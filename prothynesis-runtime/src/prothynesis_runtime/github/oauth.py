from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel


class GitHubOAuthError(Exception):
    pass


class GitHubProfile(BaseModel):
    id: int
    login: str
    name: str | None = None
    email: str | None = None
    avatar_url: str | None = None


class GitHubAuthResult(BaseModel):
    access_token: str
    token_type: str = "bearer"
    scope: str | None = None
    github_id: int | None = None
    username: str | None = None
    expires_at: float | None = None


class GitHubOAuthClient:
    AUTHORIZATION_URL = "https://github.com/login/oauth/authorize"
    TOKEN_URL = "https://github.com/login/oauth/access_token"
    USER_URL = "https://api.github.com/user"
    REVOKE_URL = "https://api.github.com/applications/{client_id}/token"
    DEFAULT_SCOPE = "read:user"

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_authorization_url(self, state: str | None = None) -> str:
        from urllib.parse import urlencode
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.DEFAULT_SCOPE,
            "state": state or "",
            "allow_signup": "true",
        }
        return f"{self.AUTHORIZATION_URL}?{urlencode(params)}"

    def exchange_code(self, code: str) -> GitHubAuthResult | None:
        try:
            resp = requests.post(
                self.TOKEN_URL,
                json={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                },
                headers={"Accept": "application/json"},
                timeout=30,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if "error" in data:
                return None
            expires_at = None
            if "expires_in" in data:
                expires_at = time.time() + int(data["expires_in"])
            return GitHubAuthResult(
                access_token=data["access_token"],
                token_type=data.get("token_type", "bearer"),
                scope=data.get("scope"),
                expires_at=expires_at,
            )
        except requests.RequestException:
            return None

    def refresh_token(self, refresh_token: str) -> GitHubAuthResult | None:
        try:
            resp = requests.post(
                self.TOKEN_URL,
                json={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                headers={"Accept": "application/json"},
                timeout=30,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if "error" in data:
                return None
            expires_at = None
            if "expires_in" in data:
                expires_at = time.time() + int(data["expires_in"])
            return GitHubAuthResult(
                access_token=data["access_token"],
                token_type=data.get("token_type", "bearer"),
                scope=data.get("scope"),
                expires_at=expires_at,
            )
        except requests.RequestException:
            return None

    def revoke_token(self, token: str) -> bool:
        try:
            resp = requests.delete(
                self.REVOKE_URL.format(client_id=self.client_id),
                json={"access_token": token},
                auth=(self.client_id, self.client_secret),
                timeout=30,
            )
            return resp.status_code in (204, 200)
        except requests.RequestException:
            return False

    def get_user_profile(self, token: str) -> GitHubProfile | None:
        try:
            resp = requests.get(
                self.USER_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=30,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return GitHubProfile(
                id=data["id"],
                login=data["login"],
                name=data.get("name"),
                email=data.get("email"),
                avatar_url=data.get("avatar_url"),
            )
        except requests.RequestException:
            return None


class GitHubAppAuth:
    def __init__(self, app_id: str, private_key_path: str | Path) -> None:
        self.app_id = app_id
        self.private_key_path = Path(private_key_path)
        self._private_key = self._load_private_key()

    def _load_private_key(self) -> str | None:
        try:
            return self.private_key_path.read_text()
        except OSError:
            return None

    def _create_jwt(self) -> str | None:
        if not self._private_key:
            return None
        try:
            import jwt
            now = int(time.time())
            payload = {
                "iat": now,
                "exp": now + (10 * 60),
                "iss": int(self.app_id),
            }
            return jwt.encode(payload, self._private_key, algorithm="RS256")
        except Exception:
            return None

    def get_installation_access_token(self, installation_id: int | str) -> str | None:
        jwt_token = self._create_jwt()
        if not jwt_token:
            return None
        try:
            resp = requests.post(
                f"https://api.github.com/app/installations/{installation_id}/access_tokens",
                headers={
                    "Authorization": f"Bearer {jwt_token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=30,
            )
            if resp.status_code != 201:
                return None
            data = resp.json()
            return data.get("token")
        except requests.RequestException:
            return None

    def get_app_installation_id(self) -> int | None:
        jwt_token = self._create_jwt()
        if not jwt_token:
            return None
        try:
            resp = requests.get(
                "https://api.github.com/app/installations",
                headers={
                    "Authorization": f"Bearer {jwt_token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=30,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data and isinstance(data, list):
                return data[0].get("id")
            return None
        except requests.RequestException:
            return None


class CredentialStore:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._key_path = self.storage_path.parent / f"{self.storage_path.name}.key"
        self._ensure_key()

    def _ensure_key(self) -> None:
        if not self._key_path.exists():
            key = Fernet.generate_key()
            self._key_path.write_bytes(key)
            try:
                os.chmod(self._key_path, 0o600)
            except OSError:
                pass

    def _get_fernet(self) -> Fernet:
        key = self._key_path.read_bytes()
        return Fernet(key)

    def save_credentials(self, auth_result: GitHubAuthResult) -> bool:
        try:
            fernet = self._get_fernet()
            data = auth_result.model_dump_json().encode("utf-8")
            encrypted = fernet.encrypt(data)
            self.storage_path.write_bytes(encrypted)
            return True
        except (OSError, Exception):
            return False

    def load_credentials(self) -> GitHubAuthResult | None:
        try:
            if not self.storage_path.exists():
                return None
            fernet = self._get_fernet()
            encrypted = self.storage_path.read_bytes()
            data = fernet.decrypt(encrypted)
            return GitHubAuthResult.model_validate_json(data.decode("utf-8"))
        except (OSError, InvalidToken, Exception):
            return None

    def clear_credentials(self) -> bool:
        try:
            if self.storage_path.exists():
                self.storage_path.unlink()
            return True
        except OSError:
            return False

    def is_authenticated(self) -> bool:
        creds = self.load_credentials()
        if not creds:
            return False
        if creds.expires_at is not None and creds.expires_at < time.time():
            self.clear_credentials()
            return False
        return True
