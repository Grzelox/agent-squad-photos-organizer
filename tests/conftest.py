import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for all tests in the session."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image_files(temp_test_dir):
    """Create sample image files for testing."""
    images = []

    # Create different types of test images
    test_images = [
        ("test_red.jpg", (100, 100), "red"),
        ("test_blue.png", (150, 100), "blue"),
        ("test_green.bmp", (200, 150), "green"),
        ("test_large.jpg", (800, 600), "yellow"),
    ]

    for filename, size, color in test_images:
        img = Image.new("RGB", size, color=color)
        image_path = temp_test_dir / filename
        img.save(image_path)
        images.append(str(image_path))

    yield images


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama response for testing."""
    return {"message": {"content": "This is a test response from the AI model."}}


@pytest.fixture
def sample_metadata_dict():
    """Sample metadata dictionary for testing."""
    return {
        "filename": "test_image.jpg",
        "filepath": "/path/to/test_image.jpg",
        "size_bytes": 1024,
        "dimensions": "100x100",
        "width": 100,
        "height": 100,
        "format": "JPEG",
        "mode": "RGB",
        "aspect_ratio": 1.0,
        "file_hash": "d41d8cd98f00b204e9800998ecf8427e",
        "created_date": "2023-01-01 12:00:00",
        "modified_date": "2023-01-01 12:00:00",
    }
