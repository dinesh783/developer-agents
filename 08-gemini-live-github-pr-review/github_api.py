"""Small GitHub REST API client for live pull-request reviews."""

import json
import os
import base64
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

API_ROOT = "https://api.github.com"
MAX_FILES = 1000
MAX_PATCH_CHARACTERS = 150_000


class GitHubAPIError(RuntimeError):
    """An actionable GitHub API request failure."""


def parse_repository(value: str) -> tuple[str, str]:
    """Validate and split an `owner/repository` identifier."""
    parts = value.strip().split("/")
    if len(parts) != 2 or any(
        not part or part in {".", ".."} or not all(char.isalnum() or char in "-_." for char in part)
        for part in parts
    ):
        raise ValueError("Repository must use the owner/repository format.")
    return parts[0], parts[1]


class GitHubClient:
    """Read pull-request data and publish one review using GitHub REST."""

    def __init__(
        self,
        repository: str,
        *,
        token: str | None = None,
    ) -> None:
        self.owner, self.name = parse_repository(repository)
        self.repository = f"{self.owner}/{self.name}"
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not self.token:
            raise RuntimeError(
                "Set GITHUB_TOKEN (or GH_TOKEN) with access to read pull requests and create reviews."
            )
        self._repo_path = f"/repos/{quote(self.owner, safe='')}/{quote(self.name, safe='')}"

    def _request(self, method: str, path: str, payload: dict[str, object] | None = None) -> object:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            f"{API_ROOT}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "gemini-pr-review-agent",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                content = response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code in (401, 403):
                raise GitHubAPIError(
                    f"GitHub denied {method} {path} (HTTP {error.code}). Check GITHUB_TOKEN "
                    f"permissions and rate limits. Details: {detail}"
                ) from error
            if error.code == 404:
                raise GitHubAPIError(
                    f"GitHub resource was not found for {method} {path}. Check the repository, PR number, "
                    f"and token access. Details: {detail}"
                ) from error
            raise GitHubAPIError(
                f"GitHub API request failed: {method} {path} returned HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise GitHubAPIError(f"Could not connect to GitHub: {error.reason}") from error

        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            raise GitHubAPIError(f"GitHub returned invalid JSON for {method} {path}.") from error

    def get_pull_request(self, pr_number: int) -> dict[str, object]:
        """Fetch one pull request and reject unexpected response shapes."""
        result = self._request("GET", f"{self._repo_path}/pulls/{pr_number}")
        if not isinstance(result, dict):
            raise GitHubAPIError("GitHub returned an invalid pull request response.")
        return result

    def list_pull_request_files(self, pr_number: int) -> list[dict[str, object]]:
        """Fetch all changed files, failing explicitly if the bounded review payload is exceeded."""
        files: list[dict[str, object]] = []
        patch_characters = 0
        page = 1
        while True:
            result = self._request(
                "GET",
                f"{self._repo_path}/pulls/{pr_number}/files?per_page=100&page={page}",
            )
            if not isinstance(result, list):
                raise GitHubAPIError("GitHub returned an invalid pull request files response.")
            if any(not isinstance(item, dict) for item in result):
                raise GitHubAPIError("GitHub returned an invalid changed-file entry.")
            files.extend(result)
            patch_characters += sum(
                len(patch)
                for item in result
                if isinstance((patch := item.get("patch")), str)
            )
            if len(files) > MAX_FILES:
                raise GitHubAPIError(
                    f"PR #{pr_number} changes more than {MAX_FILES} files; refusing a partial review."
                )
            if patch_characters > MAX_PATCH_CHARACTERS:
                raise GitHubAPIError(
                    f"PR #{pr_number} diff exceeds {MAX_PATCH_CHARACTERS} characters; refusing a partial review."
                )
            if len(result) < 100:
                return files
            page += 1

    def get_file(self, file_path: str, ref: str) -> str:
        """Fetch and decode a UTF-8 file from a specific commit."""
        encoded_path = "/".join(quote(segment, safe="") for segment in file_path.split("/"))
        result = self._request(
            "GET",
            f"{self._repo_path}/contents/{encoded_path}?ref={quote(ref, safe='')}",
        )
        if not isinstance(result, dict) or result.get("encoding") != "base64":
            raise GitHubAPIError(f"GitHub did not return base64 file content for {file_path!r}.")
        content = result.get("content")
        if not isinstance(content, str):
            raise GitHubAPIError(f"GitHub returned no file content for {file_path!r}.")
        try:
            compact_content = "".join(content.split())
            return base64.b64decode(compact_content, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise GitHubAPIError(f"Could not decode UTF-8 content for {file_path!r}.") from error

    def create_review(
        self,
        pr_number: int,
        *,
        commit_id: str,
        body: str,
    ) -> dict[str, object]:
        """Create a single top-level COMMENT review on the pull request."""
        result = self._request(
            "POST",
            f"{self._repo_path}/pulls/{pr_number}/reviews",
            {"commit_id": commit_id, "body": body, "event": "COMMENT"},
        )
        if not isinstance(result, dict) or not isinstance(result.get("html_url"), str):
            raise GitHubAPIError("GitHub did not confirm creation of the pull request review.")
        return result
