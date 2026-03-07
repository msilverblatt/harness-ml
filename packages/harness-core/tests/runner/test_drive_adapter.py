"""Tests for Google Drive adapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestDriveImportGuard:
    def test_import_error_without_deps(self):
        from harnessml.core.runner.drives.drive import _check_deps
        _check_deps()


class TestDriveAuth:
    @patch("harnessml.core.runner.drives.drive._build_service")
    def test_get_service_caches_token(self, mock_build, tmp_path):
        from harnessml.core.runner.drives.drive import get_service
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        svc = get_service(credentials_dir=tmp_path)
        assert svc is mock_service


class TestDriveUpload:
    @patch("harnessml.core.runner.drives.drive.get_service")
    def test_upload_file(self, mock_get_service, tmp_path):
        from harnessml.core.runner.drives.drive import upload_file
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2")
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_create = mock_service.files.return_value.create
        mock_create.return_value.execute.return_value = {"id": "file123", "name": "test.csv"}
        result = upload_file(test_file, credentials_dir=tmp_path)
        assert result["id"] == "file123"
        assert result["name"] == "test.csv"

    @patch("harnessml.core.runner.drives.drive.get_service")
    def test_upload_file_with_folder(self, mock_get_service, tmp_path):
        from harnessml.core.runner.drives.drive import upload_file
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2")
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_create = mock_service.files.return_value.create
        mock_create.return_value.execute.return_value = {"id": "file456", "name": "test.csv"}
        result = upload_file(test_file, folder_id="folder789", credentials_dir=tmp_path)
        assert result["id"] == "file456"

    @patch("harnessml.core.runner.drives.drive.get_service")
    def test_upload_returns_colab_url_for_ipynb(self, mock_get_service, tmp_path):
        from harnessml.core.runner.drives.drive import upload_file
        test_file = tmp_path / "notebook.ipynb"
        test_file.write_text("{}")
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_create = mock_service.files.return_value.create
        mock_create.return_value.execute.return_value = {"id": "nb123", "name": "notebook.ipynb"}
        result = upload_file(test_file, credentials_dir=tmp_path)
        assert "colab_url" in result
        assert "nb123" in result["colab_url"]


class TestDriveCreateFolder:
    @patch("harnessml.core.runner.drives.drive.get_service")
    def test_create_folder(self, mock_get_service, tmp_path):
        from harnessml.core.runner.drives.drive import create_folder
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_create = mock_service.files.return_value.create
        mock_create.return_value.execute.return_value = {"id": "folder123", "name": "my_folder"}
        result = create_folder("my_folder", credentials_dir=tmp_path)
        assert result["id"] == "folder123"


class TestDriveListFiles:
    @patch("harnessml.core.runner.drives.drive.get_service")
    def test_list_files(self, mock_get_service, tmp_path):
        from harnessml.core.runner.drives.drive import list_files
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_list = mock_service.files.return_value.list
        mock_list.return_value.execute.return_value = {
            "files": [{"id": "f1", "name": "file1.csv"}, {"id": "f2", "name": "file2.csv"}]
        }
        result = list_files(credentials_dir=tmp_path)
        assert len(result) == 2
        assert result[0]["id"] == "f1"
