import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from datetime import datetime

from services.organization_service import OrganizationService


class TestOrganizationService:
    """Test cases for OrganizationService class."""

    @pytest.fixture
    def temp_photos_dir(self):
        """Create a temporary directory for test photos."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def org_service(self, temp_photos_dir):
        """Create an OrganizationService instance with temporary directory."""
        return OrganizationService(str(temp_photos_dir))

    @pytest.fixture
    def sample_metadata_df(self):
        """Create a sample metadata DataFrame for testing."""
        return pd.DataFrame({
            "filename": ["photo1.jpg", "photo2.jpg", "photo3.jpg", "photo4.jpg"],
            "filepath": ["/path/photo1.jpg", "/path/photo2.jpg", "/path/photo3.jpg", "/path/photo4.jpg"],
            "size_bytes": [1024, 2048, 1024, 3072],
            "width": [1920, 1080, 1920, 800],
            "height": [1080, 1920, 1080, 600],
            "aspect_ratio": [1.78, 0.56, 1.78, 1.33],
            "file_hash": ["hash1", "hash2", "hash1", "hash3"],  # hash1 is duplicated
            "created_date": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4)
            ],
            "exif_Make": ["Canon", "Nikon", "Canon", "Sony"],
            "exif_Model": ["EOS R5", "D850", "EOS R5", "A7III"]
        })

    def test_init_default_directory(self):
        """Test OrganizationService initialization with default directory."""
        service = OrganizationService()
        assert service.photos_directory == Path("photos")
        assert service.organized_directory == Path("photos") / "organized"

    def test_init_custom_directory(self, temp_photos_dir):
        """Test OrganizationService initialization with custom directory."""
        service = OrganizationService(str(temp_photos_dir))
        assert service.photos_directory == temp_photos_dir
        assert service.organized_directory == temp_photos_dir / "organized"

    def test_find_similar_photos_empty_df(self, org_service):
        """Test finding similar photos with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = org_service.find_similar_photos(empty_df)
        assert result == []

    def test_find_similar_photos_insufficient_columns(self, org_service):
        """Test finding similar photos with insufficient metadata columns."""
        df = pd.DataFrame({"filename": ["photo1.jpg"], "size_bytes": [1024]})
        result = org_service.find_similar_photos(df)
        assert result == []

    def test_find_similar_photos_success(self, org_service, sample_metadata_df):
        """Test finding similar photos successfully."""
        result = org_service.find_similar_photos(sample_metadata_df, similarity_threshold=0.5)
        
        assert isinstance(result, list)
        # Should find similar photos based on metadata
        assert len(result) > 0
        
        for pair in result:
            assert "photo1" in pair
            assert "photo2" in pair
            assert "similarity" in pair
            assert "path1" in pair
            assert "path2" in pair
            assert pair["similarity"] > 0.5

    def test_find_similar_photos_high_threshold(self, org_service, sample_metadata_df):
        """Test finding similar photos with high threshold."""
        result = org_service.find_similar_photos(sample_metadata_df, similarity_threshold=0.99)
        # With high threshold, should find fewer or no similar photos
        assert isinstance(result, list)

    def test_cluster_photos_empty_df(self, org_service):
        """Test clustering photos with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = org_service.cluster_photos(empty_df)
        assert result == {}

    def test_cluster_photos_insufficient_columns(self, org_service):
        """Test clustering photos with insufficient metadata columns."""
        df = pd.DataFrame({"filename": ["photo1.jpg"], "size_bytes": [1024]})
        result = org_service.cluster_photos(df)
        assert "error" in result
        assert "Insufficient metadata" in result["error"]

    def test_cluster_photos_success(self, org_service, sample_metadata_df):
        """Test clustering photos successfully."""
        result = org_service.cluster_photos(sample_metadata_df, n_clusters=2)
        
        assert "error" not in result
        assert len(result) == 2
        
        for cluster_name, cluster_info in result.items():
            assert "count" in cluster_info
            assert "photos" in cluster_info
            assert "avg_dimensions" in cluster_info
            assert "avg_size_mb" in cluster_info
            assert isinstance(cluster_info["count"], int)
            assert isinstance(cluster_info["photos"], list)
            assert isinstance(cluster_info["avg_size_mb"], float)

    def test_cluster_photos_exception(self, org_service):
        """Test clustering photos when exception occurs."""
        # Create a DataFrame that will cause clustering to fail
        df = pd.DataFrame({
            "filename": ["photo1.jpg"],
            "width": [np.nan],  # NaN values will cause clustering to fail
            "height": [np.nan],
            "size_bytes": [1024],
            "aspect_ratio": [np.nan]
        })
        
        result = org_service.cluster_photos(df)
        assert "error" in result

    def test_find_duplicates_empty_df(self, org_service):
        """Test finding duplicates with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = org_service.find_duplicates(empty_df)
        assert result == {}

    def test_find_duplicates_success(self, org_service, sample_metadata_df):
        """Test finding duplicates successfully."""
        result = org_service.find_duplicates(sample_metadata_df)
        
        assert isinstance(result, dict)
        # Should find duplicates based on file_hash
        assert len(result) > 0
        
        for hash_val, group_info in result.items():
            assert "files" in group_info
            assert "size_mb" in group_info
            assert isinstance(group_info["files"], list)
            assert len(group_info["files"]) > 1  # Duplicates should have multiple files

    def test_find_duplicates_no_duplicates(self, org_service):
        """Test finding duplicates when no duplicates exist."""
        df = pd.DataFrame({
            "filename": ["photo1.jpg", "photo2.jpg"],
            "filepath": ["/path/photo1.jpg", "/path/photo2.jpg"],
            "file_hash": ["hash1", "hash2"],
            "size_bytes": [1024, 2048]
        })
        
        result = org_service.find_duplicates(df)
        assert result == {}

    def test_organize_by_date_dry_run(self, org_service, sample_metadata_df):
        """Test organizing by date in dry run mode."""
        result = org_service.organize_by_date(sample_metadata_df, dry_run=True)
        
        assert result["dry_run"] is True
        assert "plan" in result
        assert "moved_files" in result
        assert "errors" in result
        
        # Check that plan contains expected structure
        for filename, info in result["plan"].items():
            assert "target" in info
            assert "source" in info

    def test_organize_by_date_execute(self, org_service, sample_metadata_df, temp_photos_dir):
        """Test organizing by date in execute mode."""
        # Create the organized directory
        organized_dir = temp_photos_dir / "organized"
        organized_dir.mkdir(exist_ok=True)
        
        # Create some test files
        for filename in sample_metadata_df["filename"]:
            (temp_photos_dir / filename).write_text("test content")
        
        result = org_service.organize_by_date(sample_metadata_df, dry_run=False)
        
        assert result["dry_run"] is False
        assert "moved_files" in result
        assert "errors" in result

    def test_organize_by_camera_dry_run(self, org_service, sample_metadata_df):
        """Test organizing by camera in dry run mode."""
        result = org_service.organize_by_camera(sample_metadata_df, dry_run=True)
        
        assert result["dry_run"] is True
        assert "plan" in result
        assert "moved_files" in result
        assert "errors" in result

    def test_organize_by_camera_execute(self, org_service, sample_metadata_df, temp_photos_dir):
        """Test organizing by camera in execute mode."""
        # Create the organized directory
        organized_dir = temp_photos_dir / "organized"
        organized_dir.mkdir(exist_ok=True)
        
        # Create some test files
        for filename in sample_metadata_df["filename"]:
            (temp_photos_dir / filename).write_text("test content")
        
        result = org_service.organize_by_camera(sample_metadata_df, dry_run=False)
        
        assert result["dry_run"] is False
        assert "moved_files" in result
        assert "errors" in result

    def test_get_collection_stats_empty_df(self, org_service):
        """Test getting collection stats with empty DataFrame."""
        empty_df = pd.DataFrame()
        stats = org_service.get_collection_stats(empty_df)
        
        assert stats == {}

    def test_get_collection_stats_success(self, org_service, sample_metadata_df):
        """Test getting collection stats successfully."""
        # Add missing columns to the sample DataFrame
        sample_metadata_df["format"] = ["JPEG", "JPEG", "JPEG", "JPEG"]
        sample_metadata_df["dimensions"] = ["1920x1080", "1080x1920", "1920x1080", "800x600"]
        
        stats = org_service.get_collection_stats(sample_metadata_df)
        
        assert stats["total_photos"] == 4
        assert stats["total_size_mb"] == pytest.approx(7.0 / 1024, rel=1e-2)  # 7168 bytes
        assert stats["avg_file_size_mb"] == pytest.approx(1.75 / 1024, rel=1e-2)
        assert "2023-01-01" in stats["date_range"]
        assert "2023-01-04" in stats["date_range"]
        assert "formats" in stats
        assert "cameras" in stats

    def test_get_collection_stats_with_formats(self, org_service):
        """Test getting collection stats with format information."""
        df = pd.DataFrame({
            "filename": ["photo1.jpg", "photo2.png", "photo3.jpg"],
            "size_bytes": [1024, 2048, 3072],
            "format": ["JPEG", "PNG", "JPEG"],
            "dimensions": ["1920x1080", "1080x1920", "800x600"],
            "aspect_ratio": [1.78, 0.56, 1.33],
            "created_date": [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
        })
        
        stats = org_service.get_collection_stats(df)
        assert "formats" in stats

    def test_get_collection_stats_with_cameras(self, org_service):
        """Test getting collection stats with camera information."""
        df = pd.DataFrame({
            "filename": ["photo1.jpg", "photo2.jpg"],
            "size_bytes": [1024, 2048],
            "format": ["JPEG", "JPEG"],
            "dimensions": ["1920x1080", "1080x1920"],
            "aspect_ratio": [1.78, 0.56],
            "created_date": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "exif_Make": ["Canon", "Nikon"],
            "exif_Model": ["EOS R5", "D850"]
        })
        
        stats = org_service.get_collection_stats(df)
        assert "cameras" in stats
        assert len(stats["cameras"]) > 0

    def test_clean_folder_name(self, org_service):
        """Test folder name cleaning."""
        test_cases = [
            ("Canon EOS R5", "Canon_EOS_R5"),
            ("Nikon D850", "Nikon_D850"),
            ("Sony A7III", "Sony_A7III"),
            ("2023-01-01", "2023-01-01"),
            ("Folder/With/Slashes", "Folder_With_Slashes"),
            ("File with spaces", "File_with_spaces"),
            ("File.with.dots", "File.with.dots"),  # Dots are not in invalid_chars
            ("File@with#special$chars", "File@with#special$chars"),  # These chars are not in invalid_chars
        ]
        
        for input_name, expected_output in test_cases:
            result = org_service._clean_folder_name(input_name)
            assert result == expected_output

    def test_organize_by_date_missing_dates(self, org_service):
        """Test organizing by date when dates are missing."""
        df = pd.DataFrame({
            "filename": ["photo1.jpg", "photo2.jpg"],
            "size_bytes": [1024, 2048],
            # Missing created_date column
        })
        
        result = org_service.organize_by_date(df, dry_run=True)
        assert "error" in result

    def test_organize_by_camera_missing_camera_info(self, org_service):
        """Test organizing by camera when camera info is missing."""
        df = pd.DataFrame({
            "filename": ["photo1.jpg", "photo2.jpg"],
            "size_bytes": [1024, 2048],
            # Missing camera columns
        })
        
        result = org_service.organize_by_camera(df, dry_run=True)
        assert "error" in result

    @patch('services.organization_service.shutil.move')
    def test_organize_by_date_file_move_error(self, mock_move, org_service, sample_metadata_df, temp_photos_dir):
        """Test organizing by date when file move fails."""
        # Create the organized directory
        organized_dir = temp_photos_dir / "organized"
        organized_dir.mkdir(exist_ok=True)
        
        # Create some test files
        for filename in sample_metadata_df["filename"]:
            (temp_photos_dir / filename).write_text("test content")
        
        # Make shutil.move raise an exception
        mock_move.side_effect = Exception("Move failed")
        
        result = org_service.organize_by_date(sample_metadata_df, dry_run=False)
        
        assert result["dry_run"] is False
        assert "errors" in result
        assert len(result["errors"]) > 0

    def test_cluster_photos_different_cluster_counts(self, org_service, sample_metadata_df):
        """Test clustering with different numbers of clusters."""
        for n_clusters in [1, 2, 3, 4]:
            result = org_service.cluster_photos(sample_metadata_df, n_clusters=n_clusters)
            assert "error" not in result
            assert len(result) == n_clusters

    def test_find_similar_photos_different_thresholds(self, org_service, sample_metadata_df):
        """Test finding similar photos with different thresholds."""
        thresholds = [0.1, 0.5, 0.9, 0.99]
        
        for threshold in thresholds:
            result = org_service.find_similar_photos(sample_metadata_df, similarity_threshold=threshold)
            assert isinstance(result, list)
            
            # Higher thresholds should generally find fewer similar pairs
            for pair in result:
                assert pair["similarity"] > threshold 