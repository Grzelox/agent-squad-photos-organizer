import pytest
from unittest.mock import Mock, patch
import pandas as pd
from modules.config import config
from modules.services.ai_service import AIService


class TestAIService:
    """Test cases for AIService class."""

    @pytest.fixture
    def ai_service(self):
        """Create an AIService instance for testing."""
        return AIService("test_model")

    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama response structure."""
        return {"message": {"content": "This is a test response from the AI model."}}

    def test_init_default_model(self):
        """Test AIService initialization with default model."""
        service = AIService()
        assert service.model_name == config.model_name

    def test_init_custom_model(self):
        """Test AIService initialization with custom model."""
        service = AIService("custom_model:latest")
        assert service.model_name == "custom_model:latest"

    @patch("modules.services.ai_service.ollama.list")
    def test_is_ollama_available_success(self, mock_ollama_list, ai_service):
        """Test successful Ollama availability check."""
        mock_ollama_list.return_value = {"models": []}
        assert ai_service.is_ollama_available() is True

    @patch("modules.services.ai_service.ollama.list")
    def test_is_ollama_available_failure(self, mock_ollama_list, ai_service):
        """Test failed Ollama availability check."""
        mock_ollama_list.side_effect = Exception("Connection failed")
        assert ai_service.is_ollama_available() is False

    @patch("modules.services.ai_service.ollama.list")
    def test_get_available_models_dict_response(self, mock_ollama_list, ai_service):
        """Test getting available models with dict response."""
        mock_response = {
            "models": [{"name": "model1"}, {"name": "model2"}, {"name": "model3"}]
        }
        mock_ollama_list.return_value = mock_response

        models = ai_service.get_available_models()
        assert models == ["model1", "model2", "model3"]

    @patch("modules.services.ai_service.ollama.list")
    def test_get_available_models_list_response(self, mock_ollama_list, ai_service):
        """Test getting available models with list response."""
        mock_response = [{"name": "model1"}, {"name": "model2"}]
        mock_ollama_list.return_value = mock_response

        models = ai_service.get_available_models()
        assert models == ["model1", "model2"]

    @patch("modules.services.ai_service.ollama.list")
    def test_get_available_models_objects_response(self, mock_ollama_list, ai_service):
        """Test getting available models with objects response."""
        mock_model1 = Mock()
        mock_model1.model = "model1:latest"
        mock_model2 = Mock()
        mock_model2.model = "model2"

        mock_response = Mock()
        mock_response.models = [mock_model1, mock_model2]
        mock_ollama_list.return_value = mock_response

        models = ai_service.get_available_models()
        assert models == ["model1", "model2"]

    @patch("modules.services.ai_service.ollama.list")
    def test_get_available_models_empty_response(self, mock_ollama_list, ai_service):
        """Test getting available models with empty response."""
        mock_ollama_list.return_value = {}
        models = ai_service.get_available_models()
        assert models == []

    @patch("modules.services.ai_service.ollama.list")
    def test_get_available_models_exception(self, mock_ollama_list, ai_service):
        """Test getting available models when exception occurs."""
        mock_ollama_list.side_effect = Exception("Test error")
        models = ai_service.get_available_models()
        assert models == []

    @patch("modules.services.ai_service.ollama.chat")
    def test_analyze_image_with_text_success(
        self, mock_ollama_chat, ai_service, mock_ollama_response
    ):
        """Test successful image analysis with text."""
        mock_ollama_chat.return_value = mock_ollama_response

        result = ai_service.analyze_image_with_text(
            "Describe this image", "base64_encoded_image_data"
        )

        assert result == "This is a test response from the AI model."
        mock_ollama_chat.assert_called_once()

    @patch("modules.services.ai_service.ollama.chat")
    def test_analyze_image_with_text_connection_error(
        self, mock_ollama_chat, ai_service
    ):
        """Test image analysis with connection error."""
        mock_ollama_chat.side_effect = Exception("Connection refused")

        result = ai_service.analyze_image_with_text(
            "Describe this image", "base64_encoded_image_data"
        )

        assert "Error analyzing image" in result
        assert "Make sure Ollama is running" in result

    @patch("modules.services.ai_service.ollama.chat")
    def test_analyze_image_with_text_model_not_found(
        self, mock_ollama_chat, ai_service
    ):
        """Test image analysis with model not found error."""
        mock_ollama_chat.side_effect = Exception("Model test_model not found")

        result = ai_service.analyze_image_with_text(
            "Describe this image", "base64_encoded_image_data"
        )

        assert "Error analyzing image" in result
        assert "Model test_model not found" in result

    @patch("modules.services.ai_service.ollama.chat")
    def test_analyze_multiple_images_success(
        self, mock_ollama_chat, ai_service, mock_ollama_response
    ):
        """Test successful multiple image analysis."""
        mock_ollama_chat.return_value = mock_ollama_response

        result = ai_service.analyze_multiple_images(
            "Compare these images", ["base64_image1", "base64_image2"]
        )

        assert result == "This is a test response from the AI model."
        mock_ollama_chat.assert_called_once()

    @patch("modules.services.ai_service.ollama.chat")
    def test_analyze_multiple_images_error(self, mock_ollama_chat, ai_service):
        """Test multiple image analysis with error."""
        mock_ollama_chat.side_effect = Exception("Test error")

        result = ai_service.analyze_multiple_images(
            "Compare these images", ["base64_image1", "base64_image2"]
        )

        assert "Error analyzing multiple images" in result

    @patch("modules.services.ai_service.ollama.chat")
    def test_compare_images_with_prompt(
        self, mock_ollama_chat, ai_service, mock_ollama_response
    ):
        """Test image comparison with custom prompt."""
        mock_ollama_chat.return_value = mock_ollama_response

        result = ai_service.compare_images(
            "base64_image1", "base64_image2", "Custom comparison prompt"
        )

        assert result == "This is a test response from the AI model."
        mock_ollama_chat.assert_called_once()

    @patch("modules.services.ai_service.ollama.chat")
    def test_compare_images_default_prompt(
        self, mock_ollama_chat, ai_service, mock_ollama_response
    ):
        """Test image comparison with default prompt."""
        mock_ollama_chat.return_value = mock_ollama_response

        result = ai_service.compare_images("base64_image1", "base64_image2")

        assert result == "This is a test response from the AI model."
        # Check that the default prompt was used
        call_args = mock_ollama_chat.call_args
        assert "Compare these two images" in call_args[1]["messages"][0]["content"]

    @patch("modules.services.ai_service.ollama.chat")
    def test_analyze_photos_metadata_success(
        self, mock_ollama_chat, ai_service, mock_ollama_response
    ):
        """Test successful photo metadata analysis."""
        mock_ollama_chat.return_value = mock_ollama_response

        metadata_summary = "Total photos: 100, Date range: 2020-2023"
        result = ai_service.analyze_photos_metadata(metadata_summary)

        assert result == "This is a test response from the AI model."
        mock_ollama_chat.assert_called_once()

    @patch("modules.services.ai_service.ollama.chat")
    def test_analyze_photos_metadata_error(self, mock_ollama_chat, ai_service):
        """Test photo metadata analysis with error."""
        mock_ollama_chat.side_effect = Exception("Test error")

        result = ai_service.analyze_photos_metadata("Test metadata")
        assert "Error analyzing metadata" in result

    @patch("modules.services.ai_service.ollama.chat")
    def test_suggest_organization_strategy_success(
        self, mock_ollama_chat, ai_service, mock_ollama_response
    ):
        """Test successful organization strategy suggestion."""
        mock_ollama_chat.return_value = mock_ollama_response

        photo_stats = {
            "total_photos": 100,
            "date_range": "2020-2023",
            "unique_cameras": 3,
            "duplicate_groups": 5,
            "avg_file_size": 2.5,
            "common_resolution": "1920x1080",
        }

        result = ai_service.suggest_organization_strategy(photo_stats)
        assert result == "This is a test response from the AI model."
        mock_ollama_chat.assert_called_once()

    @patch("modules.services.ai_service.ollama.chat")
    def test_suggest_organization_strategy_error(self, mock_ollama_chat, ai_service):
        """Test organization strategy suggestion with error."""
        mock_ollama_chat.side_effect = Exception("Test error")

        result = ai_service.suggest_organization_strategy({})
        assert "Error suggesting organization strategy" in result

    @patch("modules.services.ai_service.ollama.chat")
    def test_answer_photo_question_success(
        self, mock_ollama_chat, ai_service, mock_ollama_response
    ):
        """Test successful photo question answering."""
        mock_ollama_chat.return_value = mock_ollama_response

        ai_service.cache.get = lambda key: None

        result = ai_service.answer_photo_question(
            "How many photos do I have?", "You have 100 photos from 2020-2023"
        )

        assert result == "This is a test response from the AI model."
        mock_ollama_chat.assert_called_once()

    @patch("modules.services.ai_service.ollama.chat")
    def test_answer_photo_question_error(self, mock_ollama_chat, ai_service):
        """Test photo question answering with error."""
        mock_ollama_chat.side_effect = Exception("Test error")

        result = ai_service.answer_photo_question("Test question", "Test context")
        assert "Error answering question" in result

    @patch("modules.services.ai_service.ollama.chat")
    def test_generate_organization_plan_success(
        self, mock_ollama_chat, ai_service, mock_ollama_response
    ):
        """Test successful organization plan generation."""
        mock_ollama_chat.return_value = mock_ollama_response

        # Create a mock DataFrame with required columns
        metadata_df = pd.DataFrame(
            {
                "filename": ["photo1.jpg", "photo2.jpg"],
                "size_bytes": [1024, 2048],
                "width": [1920, 1080],
                "height": [1080, 1920],
                "format": ["JPEG", "PNG"],
                "dimensions": ["1920x1080", "1080x1920"],
                "created_date": ["2023-01-01", "2023-01-02"],
            }
        )

        result = ai_service.generate_organization_plan("By date", metadata_df)
        assert result == "This is a test response from the AI model."
        mock_ollama_chat.assert_called_once()

    @patch("modules.services.ai_service.ollama.chat")
    def test_generate_organization_plan_error(self, mock_ollama_chat, ai_service):
        """Test organization plan generation with error."""
        mock_ollama_chat.side_effect = Exception("Test error")

        metadata_df = pd.DataFrame(
            {
                "filename": ["test.jpg"],
                "size_bytes": [1024],
                "format": ["JPEG"],
                "dimensions": ["100x100"],
                "created_date": ["2023-01-01"],
            }
        )
        result = ai_service.generate_organization_plan("Test criteria", metadata_df)
        assert "Error generating organization plan" in result

    def test_create_metadata_summary(self, ai_service):
        """Test metadata summary creation."""
        metadata_df = pd.DataFrame(
            {
                "filename": ["photo1.jpg", "photo2.jpg", "photo3.jpg"],
                "size_bytes": [1024, 2048, 3072],
                "width": [1920, 1080, 800],
                "height": [1080, 1920, 600],
                "format": ["JPEG", "PNG", "JPEG"],
                "dimensions": ["1920x1080", "1080x1920", "800x600"],
                "created_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            }
        )

        summary = ai_service._create_metadata_summary(metadata_df)

        assert "Total photos: 3" in summary
        assert "Date range: 2023-01-01 to 2023-01-03" in summary
        assert "JPEG" in summary
        assert "PNG" in summary

    def test_create_metadata_summary_empty_df(self, ai_service):
        """Test metadata summary creation with empty DataFrame."""
        metadata_df = pd.DataFrame()
        summary = ai_service._create_metadata_summary(metadata_df)
        assert "No photos found." in summary

    @patch("modules.services.ai_service.ollama.list")
    @patch("modules.services.ai_service.ollama.pull")
    def test_ensure_model_available_model_exists(
        self, mock_pull, mock_list, ai_service
    ):
        """Test model availability check when model already exists."""
        mock_list.return_value = {"models": [{"name": "test_model"}]}

        ai_service._ensure_model_available()

        mock_pull.assert_not_called()

    @patch("modules.services.ai_service.ollama.list")
    @patch("modules.services.ai_service.ollama.pull")
    def test_ensure_model_available_model_missing(
        self, mock_pull, mock_list, ai_service
    ):
        """Test model availability check when model needs to be pulled."""
        mock_list.return_value = {"models": [{"name": "other_model"}]}
        mock_pull.return_value = None

        ai_service._ensure_model_available()

        mock_pull.assert_called_once_with("test_model")

    @patch("modules.services.ai_service.ollama.list")
    @patch("modules.services.ai_service.ollama.pull")
    def test_ensure_model_available_pull_failure(
        self, mock_pull, mock_list, ai_service
    ):
        """Test model availability check when pull fails."""
        mock_list.return_value = {"models": [{"name": "other_model"}]}
        mock_pull.side_effect = Exception("Pull failed")

        # Should not raise exception
        ai_service._ensure_model_available()

    @patch("modules.services.ai_service.ollama.list")
    def test_ensure_model_available_list_failure(self, mock_list, ai_service):
        """Test model availability check when list fails."""
        mock_list.side_effect = Exception("List failed")

        # Should not raise exception
        ai_service._ensure_model_available()
