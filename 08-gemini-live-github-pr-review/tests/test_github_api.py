import base64
import json
import unittest
from unittest.mock import patch

from github_api import GitHubClient, parse_repository


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class GitHubClientTests(unittest.TestCase):
    def test_repository_identifier_is_validated(self) -> None:
        self.assertEqual(parse_repository("example/project"), ("example", "project"))
        with self.assertRaises(ValueError):
            parse_repository("https://github.com/example/project")

    def test_fetches_pr_and_all_changed_files(self) -> None:
        pr = {"number": 12, "title": "Example"}
        files = [{"filename": "src/app.py", "patch": "@@ -1 +1 @@\n-old\n+new"}]
        with patch("github_api.urlopen", side_effect=[FakeResponse(pr), FakeResponse(files)]) as request:
            client = GitHubClient("example/project", token="test-token")

            self.assertEqual(client.get_pull_request(12), pr)
            self.assertEqual(client.list_pull_request_files(12), files)

        self.assertEqual(request.call_count, 2)
        first_request = request.call_args_list[0].args[0]
        self.assertEqual(first_request.full_url, "https://api.github.com/repos/example/project/pulls/12")
        self.assertEqual(first_request.get_header("Authorization"), "Bearer test-token")

    def test_posts_a_top_level_comment_review(self) -> None:
        payload = {"html_url": "https://github.com/example/project/pull/12#pullrequestreview-1"}
        with patch("github_api.urlopen", return_value=FakeResponse(payload)) as request:
            client = GitHubClient("example/project", token="test-token")
            result = client.create_review(12, commit_id="abc123", body="Review summary")

        self.assertEqual(result, payload)
        outgoing = request.call_args.args[0]
        self.assertEqual(outgoing.method, "POST")
        self.assertEqual(
            json.loads(outgoing.data),
            {"commit_id": "abc123", "body": "Review summary", "event": "COMMENT"},
        )

    def test_reads_base64_file_content(self) -> None:
        content = base64.b64encode(b"def hello():\n    pass\n").decode("ascii")
        with patch(
            "github_api.urlopen",
            return_value=FakeResponse({"encoding": "base64", "content": content}),
        ):
            client = GitHubClient("example/project", token="test-token")

            self.assertEqual(client.get_file("src/app.py", "abc123"), "def hello():\n    pass\n")

    def test_missing_token_fails_explicitly(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GITHUB_TOKEN"):
                GitHubClient("example/project")


if __name__ == "__main__":
    unittest.main()
