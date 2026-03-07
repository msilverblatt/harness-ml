"""Google Drive adapter for uploading files and managing folders."""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_FILE = "drive_token.json"
COLAB_URL_TEMPLATE = "https://colab.research.google.com/drive/{file_id}"


def _check_deps() -> None:
    """Verify Google API dependencies are installed."""
    try:
        import google.auth  # noqa: F401
        import google_auth_oauthlib.flow  # noqa: F401
        import googleapiclient.discovery  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Google Drive integration requires google-api-python-client and "
            "google-auth-oauthlib. Install with: "
            "pip install google-api-python-client google-auth-oauthlib"
        ) from exc


def _build_service(credentials_dir: Path) -> Any:
    """Build authenticated Drive v3 service.

    Checks for cached token at credentials_dir/drive_token.json.
    If no valid token, looks for client_secret.json and runs
    InstalledAppFlow.run_local_server(port=0). Caches token.
    """
    _check_deps()
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    token_path = credentials_dir / TOKEN_FILE
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_secret_path = credentials_dir / "client_secret.json"
            if not client_secret_path.exists():
                raise FileNotFoundError(
                    f"No client_secret.json found in {credentials_dir}. "
                    "Download OAuth credentials from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secret_path), SCOPES
            )
            creds = flow.run_local_server(port=0)

        token_path.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def get_service(*, credentials_dir: Path | None = None) -> Any:
    """Public wrapper to get an authenticated Drive service.

    Defaults credentials_dir to ~/.harnessml/. Creates dir if needed.
    """
    if credentials_dir is None:
        credentials_dir = Path.home() / ".harnessml"
    credentials_dir = Path(credentials_dir)
    credentials_dir.mkdir(parents=True, exist_ok=True)
    return _build_service(credentials_dir)


def upload_file(
    file_path: str | Path,
    *,
    folder_id: str | None = None,
    name: str | None = None,
    credentials_dir: Path | None = None,
) -> dict:
    """Upload a file to Google Drive.

    Uses MediaFileUpload with auto MIME detection. If the file is .ipynb,
    adds a 'colab_url' key to the result.
    """
    from googleapiclient.http import MediaFileUpload

    file_path = Path(file_path)
    service = get_service(credentials_dir=credentials_dir)

    file_name = name or file_path.name
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type is None:
        mime_type = "application/octet-stream"

    file_metadata: dict[str, Any] = {"name": file_name}
    if folder_id is not None:
        file_metadata["parents"] = [folder_id]

    media = MediaFileUpload(str(file_path), mimetype=mime_type)
    result = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id,name")
        .execute()
    )

    if file_path.suffix == ".ipynb":
        result["colab_url"] = COLAB_URL_TEMPLATE.format(file_id=result["id"])

    return result


def create_folder(
    name: str,
    *,
    parent_id: str | None = None,
    credentials_dir: Path | None = None,
) -> dict:
    """Create a folder on Google Drive."""
    service = get_service(credentials_dir=credentials_dir)

    file_metadata: dict[str, Any] = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id is not None:
        file_metadata["parents"] = [parent_id]

    return (
        service.files()
        .create(body=file_metadata, fields="id,name")
        .execute()
    )


def list_files(
    *,
    folder_id: str | None = None,
    credentials_dir: Path | None = None,
) -> list[dict]:
    """List files on Google Drive, optionally filtered to a folder."""
    service = get_service(credentials_dir=credentials_dir)

    query = None
    if folder_id is not None:
        query = f"'{folder_id}' in parents"

    response = (
        service.files()
        .list(q=query, fields="files(id,name,mimeType)")
        .execute()
    )
    return response.get("files", [])
