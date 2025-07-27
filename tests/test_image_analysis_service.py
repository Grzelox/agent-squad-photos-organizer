import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
import tempfile
import shutil
import base64
import json
from PIL import Image

from modules.services.image_analysis_service import ImageAnalysisService


class TestImageAnalysisService:
    """Test cases for ImageAnalysisService class."""

    @pytest.fixture
    def temp_photos_dir(self):
        """Create a temporary directory for test photos."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def image_analysis_service(self, temp_photos_dir):
        """Create an ImageAnalysisService instance with temporary directory."""
        return ImageAnalysisService(str(temp_photos_dir))

    @pytest.fixture
    def sample_image_path(self, temp_photos_dir):
        """Create a sample test image."""
        img = Image.new("RGB", (100, 100), color="red")
        image_path = temp_photos_dir / "test_image.jpg"
        img.save(image_path)
        return str(image_path)

    @pytest.fixture
    def sample_metadata_df(self):
        """Create a sample metadata DataFrame for testing."""
        return pd.DataFrame(
            {
                "filename": ["photo1.jpg", "photo2.jpg", "photo3.jpg"],
                "filepath": [
                    "/path/photo1.jpg",
                    "/path/photo2.jpg",
                    "/path/photo3.jpg",
                ],
                "size_bytes": [1024, 2048, 3072],
                "width": [1920, 1080, 800],
                "height": [1080, 1920, 600],
            }
        )

    def test_init(self, temp_photos_dir):
        """Test ImageAnalysisService initialization."""
        service = ImageAnalysisService(str(temp_photos_dir))
        assert service.photos_directory == str(temp_photos_dir)
        assert service.supported_formats == {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".webp",
        }

    def test_encode_image_to_base64_success(
        self, image_analysis_service, sample_image_path
    ):
        """Test successful image encoding to base64."""
        result = image_analysis_service.encode_image_to_base64(sample_image_path)

        assert result is not None
        assert isinstance(result, str)
        # Verify it's valid base64
        try:
            decoded = base64.b64decode(result)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("Result is not valid base64")

    def test_encode_image_to_base64_nonexistent_file(self, image_analysis_service):
        """Test image encoding with non-existent file."""
        result = image_analysis_service.encode_image_to_base64("/nonexistent/image.jpg")
        assert result is None

    def test_encode_image_to_base64_invalid_file(
        self, image_analysis_service, temp_photos_dir
    ):
        """Test image encoding with invalid file."""
        # Create a text file instead of an image
        text_file = temp_photos_dir / "not_an_image.txt"
        text_file.write_text("This is not an image")

        result = image_analysis_service.encode_image_to_base64(str(text_file))
        # Should still work as it just reads bytes
        assert result is not None

    def test_get_image_hash_success(self, image_analysis_service, sample_image_path):
        """Test successful image hash generation."""
        result = image_analysis_service.get_image_hash(sample_image_path)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_image_hash_nonexistent_file(self, image_analysis_service):
        """Test image hash generation with non-existent file."""
        result = image_analysis_service.get_image_hash("/nonexistent/image.jpg")
        assert result is None

    def test_get_image_hash_invalid_file(self, image_analysis_service, temp_photos_dir):
        """Test image hash generation with invalid file."""
        # Create a text file instead of an image
        text_file = temp_photos_dir / "not_an_image.txt"
        text_file.write_text("This is not an image")

        result = image_analysis_service.get_image_hash(str(text_file))
        assert result is None

    def test_calculate_image_similarity_same_hash(self, image_analysis_service):
        """Test similarity calculation with identical hashes."""
        hash_value = "0000000000000000"  # Example hash
        similarity = image_analysis_service.calculate_image_similarity(
            hash_value, hash_value
        )
        assert similarity == 1.0

    def test_calculate_image_similarity_different_hashes(self, image_analysis_service):
        """Test similarity calculation with different hashes."""
        hash1 = "0000000000000000"
        hash2 = "ffffffffffffffff"  # Completely different
        similarity = image_analysis_service.calculate_image_similarity(hash1, hash2)
        assert 0.0 <= similarity <= 1.0

    def test_calculate_image_similarity_invalid_hash(self, image_analysis_service):
        """Test similarity calculation with invalid hash."""
        result = image_analysis_service.calculate_image_similarity(
            "invalid", "also_invalid"
        )
        assert result == 0.0

    @patch("modules.services.image_analysis_service.cv2.imread")
    def test_extract_image_features_success(self, mock_imread, image_analysis_service):
        """Test successful image feature extraction."""
        # Mock OpenCV image
        mock_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        features = image_analysis_service.extract_image_features("/test/image.jpg")

        assert isinstance(features, dict)
        assert "width" in features
        assert "height" in features
        assert "aspect_ratio" in features
        assert "total_pixels" in features
        assert "mean_color" in features
        assert "std_color" in features
        assert "brightness" in features
        assert "brightness_std" in features
        assert "edge_density" in features
        assert "dominant_colors" in features

    @patch("modules.services.image_analysis_service.cv2.imread")
    def test_extract_image_features_none_image(
        self, mock_imread, image_analysis_service
    ):
        """Test image feature extraction with None image."""
        mock_imread.return_value = None

        features = image_analysis_service.extract_image_features("/test/image.jpg")
        assert features == {}

    @patch("modules.services.image_analysis_service.cv2.imread")
    def test_extract_image_features_exception(
        self, mock_imread, image_analysis_service
    ):
        """Test image feature extraction when exception occurs."""
        mock_imread.side_effect = Exception("OpenCV error")

        features = image_analysis_service.extract_image_features("/test/image.jpg")
        assert features == {}

    def test_find_similar_images_empty_df(self, image_analysis_service):
        """Test finding similar images with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = image_analysis_service.find_similar_images(empty_df)
        assert result == {}

    def test_find_similar_images_no_similar(
        self, image_analysis_service, sample_metadata_df
    ):
        """Test finding similar images when no similar images exist."""
        # Mock the get_image_hash method to return different hashes
        with patch.object(image_analysis_service, "get_image_hash") as mock_hash:
            mock_hash.side_effect = [f"hash{i}" for i in range(len(sample_metadata_df))]

            result = image_analysis_service.find_similar_images(
                sample_metadata_df, similarity_threshold=0.9
            )
            assert result == {}

    def test_find_similar_images_with_similar(
        self, image_analysis_service, sample_metadata_df
    ):
        """Test finding similar images when similar images exist."""
        # Mock the get_image_hash method to return some similar hashes
        with patch.object(image_analysis_service, "get_image_hash") as mock_hash:
            mock_hash.side_effect = [
                "0000000000000000",
                "0000000000000000",
                "8000000000000000",
            ]  # First two are similar

            result = image_analysis_service.find_similar_images(
                sample_metadata_df, similarity_threshold=0.8
            )
            assert isinstance(result, dict)
            assert len(result) > 0

    def test_is_supported_format(self, image_analysis_service):
        """Test supported format checking."""
        supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        unsupported_formats = [".txt", ".pdf", ".doc", ".mp4", ".avi"]

        for fmt in supported_formats:
            assert image_analysis_service._is_supported_format(f"image{fmt}")

        for fmt in unsupported_formats:
            assert not image_analysis_service._is_supported_format(f"file{fmt}")

    def test_save_analysis_results(self, image_analysis_service, temp_photos_dir):
        """Test saving analysis results to JSON file."""
        test_results = {
            "group1": {"images": ["image1.jpg", "image2.jpg"], "similarity": 0.95},
            "group2": {"images": ["image3.jpg"], "similarity": 0.87},
        }

        output_path = temp_photos_dir / "test_results.json"
        image_analysis_service.save_analysis_results(test_results, str(output_path))

        assert output_path.exists()

        # Verify the saved content
        with open(output_path, "r") as f:
            saved_data = json.load(f)

        assert saved_data == test_results

    def test_load_analysis_results(self, image_analysis_service, temp_photos_dir):
        """Test loading analysis results from JSON file."""
        test_results = {
            "group1": {"images": ["image1.jpg", "image2.jpg"], "similarity": 0.95}
        }

        # Create a test JSON file
        output_path = temp_photos_dir / "test_results.json"
        with open(output_path, "w") as f:
            json.dump(test_results, f)

        loaded_data = image_analysis_service.load_analysis_results(str(output_path))
        assert loaded_data == test_results

    def test_load_analysis_results_nonexistent_file(self, image_analysis_service):
        """Test loading analysis results from non-existent file."""
        result = image_analysis_service.load_analysis_results("/nonexistent/file.json")
        assert result == {}

    def test_load_analysis_results_invalid_json(
        self, image_analysis_service, temp_photos_dir
    ):
        """Test loading analysis results from invalid JSON file."""
        # Create a file with invalid JSON
        output_path = temp_photos_dir / "invalid.json"
        output_path.write_text("This is not valid JSON")

        result = image_analysis_service.load_analysis_results(str(output_path))
        assert result == {}

    @patch("modules.services.image_analysis_service.cv2.imread")
    def test_extract_image_features_different_sizes(
        self, mock_imread, image_analysis_service
    ):
        """Test image feature extraction with different image sizes."""
        test_sizes = [(100, 100), (200, 150), (800, 600), (1920, 1080)]

        for width, height in test_sizes:
            mock_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            mock_imread.return_value = mock_image

            features = image_analysis_service.extract_image_features("/test/image.jpg")

            assert features["width"] == width
            assert features["height"] == height
            assert features["aspect_ratio"] == width / height
            assert features["total_pixels"] == width * height

    def test_calculate_image_similarity_edge_cases(self, image_analysis_service):
        """Test similarity calculation with edge cases."""
        # Test with very similar hashes
        hash1 = "0000000000000000"
        hash2 = "0000000000000001"  # Only 1 bit different
        similarity = image_analysis_service.calculate_image_similarity(hash1, hash2)
        assert similarity > 0.9  # Should be very similar

        # Test with very different hashes
        hash3 = "0000000000000000"
        hash4 = "ffffffffffffffff"  # Completely different
        similarity = image_analysis_service.calculate_image_similarity(hash3, hash4)
        assert similarity < 0.6  # Should be less similar

    def test_find_similar_images_different_thresholds(
        self, image_analysis_service, sample_metadata_df
    ):
        """Test finding similar images with different thresholds."""
        # Mock the get_image_hash method
        with patch.object(image_analysis_service, "get_image_hash") as mock_hash:
            # Create a list that can be reused for each threshold test
            hash_values = [
                "0000000000000000",
                "0000000000000000",
                "8000000000000000",
            ]  # First two are similar
            mock_hash.side_effect = hash_values * 4  # Repeat for 4 threshold tests

            # Test with different thresholds
            thresholds = [0.5, 0.8, 0.9, 0.99]
            for threshold in thresholds:
                result = image_analysis_service.find_similar_images(
                    sample_metadata_df, similarity_threshold=threshold
                )
                assert isinstance(result, dict)

    def test_save_and_load_analysis_results_roundtrip(
        self, image_analysis_service, temp_photos_dir
    ):
        """Test saving and loading analysis results in a roundtrip."""
        original_data = {
            "similar_group_1": {
                "count": 3,
                "similarity_score": 0.95,
                "images": [
                    {"filename": "image1.jpg", "path": "/path/image1.jpg"},
                    {"filename": "image2.jpg", "path": "/path/image2.jpg"},
                    {"filename": "image3.jpg", "path": "/path/image3.jpg"},
                ],
            }
        }

        output_path = temp_photos_dir / "roundtrip_test.json"

        # Save
        image_analysis_service.save_analysis_results(original_data, str(output_path))

        # Load
        loaded_data = image_analysis_service.load_analysis_results(str(output_path))

        # Verify roundtrip
        assert loaded_data == original_data

    @patch("modules.services.image_analysis_service.cv2.imread")
    def test_extract_image_features_color_analysis(
        self, mock_imread, image_analysis_service
    ):
        """Test color analysis in image feature extraction."""
        # Create a mock image with specific colors
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_image[:, :, 0] = 255  # Red channel
        mock_imread.return_value = mock_image

        features = image_analysis_service.extract_image_features("/test/image.jpg")

        assert "mean_color" in features
        assert "std_color" in features
        assert "dominant_colors" in features

        # Check that color analysis was performed
        assert len(features["mean_color"]) == 3  # RGB
        assert len(features["dominant_colors"]) > 0

    def test_get_image_hash_different_formats(
        self, image_analysis_service, temp_photos_dir
    ):
        """Test image hash generation for different image formats."""
        formats = ["jpg", "png", "bmp"]

        for fmt in formats:
            img = Image.new("RGB", (50, 50), color="blue")
            image_path = temp_photos_dir / f"test_image.{fmt}"
            img.save(image_path)

            hash_value = image_analysis_service.get_image_hash(str(image_path))
            assert hash_value is not None
            assert isinstance(hash_value, str)
            assert len(hash_value) > 0

    def test_encode_image_to_base64_different_sizes(
        self, image_analysis_service, temp_photos_dir
    ):
        """Test base64 encoding for images of different sizes."""
        sizes = [(10, 10), (100, 100), (500, 500)]

        for width, height in sizes:
            img = Image.new("RGB", (width, height), color="green")
            image_path = temp_photos_dir / f"test_{width}x{height}.jpg"
            img.save(image_path)

            encoded = image_analysis_service.encode_image_to_base64(str(image_path))
            assert encoded is not None

            # Verify it's valid base64
            decoded = base64.b64decode(encoded)
            assert len(decoded) > 0
