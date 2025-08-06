import ollama
from typing import Dict, Optional

from modules.config import config
from modules.cache import Cache


class AIService:
    """Service for handling AI operations using Ollama."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.model_name
        self.cache = Cache()
        try:
            self._ensure_model_available()
        except Exception:
            pass

    def _ensure_model_available(self):
        """Ensure the model is available locally."""
        try:
            models_response = ollama.list()

            model_names = []
            if hasattr(models_response, "models"):
                for model in models_response.models:
                    if hasattr(model, "model") and model.model:
                        model_name = (
                            model.model.split(":")[0]
                            if ":" in model.model
                            else model.model
                        )
                        model_names.append(model_name)
            elif isinstance(models_response, dict) and "models" in models_response:
                model_names = [
                    model.get("name", "")
                    for model in models_response["models"]
                    if model.get("name")
                ]
            elif isinstance(models_response, list):
                model_names = [
                    model.get("name", "")
                    for model in models_response
                    if model.get("name")
                ]

            if self.model_name not in model_names:
                try:
                    ollama.pull(self.model_name)
                except Exception as pull_error:
                    pass

        except Exception as e:
            print(f"Error checking/pulling model: {e}")
            print("Make sure Ollama is running locally (ollama serve)")
            print("You can install Ollama from: https://ollama.ai")
            print("After installation, run: ollama pull gemma3:4b")

    def is_ollama_available(self) -> bool:
        """Check if Ollama is available and running."""
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def get_available_models(self) -> list:
        """Get list of available models."""
        try:
            models_response = ollama.list()

            if hasattr(models_response, "models"):
                model_names = []
                for model in models_response.models:
                    if hasattr(model, "model") and model.model:
                        model_name = (
                            model.model.split(":")[0]
                            if ":" in model.model
                            else model.model
                        )
                        model_names.append(model_name)
                return model_names
            elif isinstance(models_response, dict) and "models" in models_response:
                model_names = [
                    model.get("name", "")
                    for model in models_response["models"]
                    if model.get("name")
                ]
                return model_names
            elif isinstance(models_response, list):
                model_names = [
                    model.get("name", "")
                    for model in models_response
                    if model.get("name")
                ]
                return model_names
            else:
                return []
        except Exception as e:
            return []

    def _handle_ollama_error(self, e: Exception, context: str) -> str:
        """Centralized error handler for Ollama API calls."""
        error_msg = f"Error {context}: {e}"
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            error_msg += "\n💡 Make sure Ollama is running: ollama serve"
        elif "model" in str(e).lower() and "not found" in str(e).lower():
            error_msg += f"\n💡 Model {self.model_name} not found. Try: ollama pull {self.model_name}"
        return error_msg

    def analyze_image_with_text(self, prompt: str, base64_image: str) -> str:
        """Analyze an image with text prompt using the AI model."""
        try:
            message = {"role": "user", "content": prompt, "images": [base64_image]}

            response = ollama.chat(model=self.model_name, messages=[message])
            return response["message"]["content"]
        except Exception as e:
            return self._handle_ollama_error(e, "analyzing image")

    def analyze_multiple_images(self, prompt: str, base64_images: list) -> str:
        """Analyze multiple images with a text prompt."""
        try:
            message = {"role": "user", "content": prompt, "images": base64_images}

            response = ollama.chat(model=self.model_name, messages=[message])
            return response["message"]["content"]
        except Exception as e:
            return self._handle_ollama_error(e, "analyzing multiple images")

    def compare_images(
        self, image1_base64: str, image2_base64: str, comparison_prompt: str = None
    ) -> str:
        """Compare two images and provide analysis."""
        if comparison_prompt is None:
            comparison_prompt = """
            Compare these two images and provide analysis on:
            1. Similarities in content, composition, and style
            2. Differences in subjects, colors, lighting
            3. Overall similarity score (0-100%)
            4. Suggested grouping category
            
            Provide your analysis in a clear, structured format.
            """

        try:
            message = {
                "role": "user",
                "content": comparison_prompt,
                "images": [image1_base64, image2_base64],
            }

            response = ollama.chat(model=self.model_name, messages=[message])
            return response["message"]["content"]
        except Exception as e:
            return self._handle_ollama_error(e, "comparing images")

    def analyze_photos_metadata(self, metadata_summary: str) -> str:
        """Analyze photo metadata and provide insights."""
        prompt = f"""
        You are a photo organization assistant. Analyze the following photo metadata and provide insights about:
        1. Organization suggestions
        2. Duplicate detection results
        3. Quality patterns
        4. Date/time patterns
        5. Device/camera patterns
        
        Photo metadata summary:
        {metadata_summary}
        
        Provide clear, actionable recommendations for organizing these photos.
        """

        try:
            response = ollama.chat(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return self._handle_ollama_error(e, "analyzing metadata")

    def suggest_organization_strategy(self, photo_stats: Dict) -> str:
        """Suggest organization strategy based on photo statistics."""
        prompt = f"""
        Based on the following photo collection statistics, suggest the best organization strategy:
        
        Statistics:
        - Total photos: {photo_stats.get('total_photos', 0)}
        - Date range: {photo_stats.get('date_range', 'Unknown')}
        - Unique cameras: {photo_stats.get('unique_cameras', 0)}
        - Duplicate groups: {photo_stats.get('duplicate_groups', 0)}
        - Average file size: {photo_stats.get('avg_file_size', 0)} MB
        - Most common resolution: {photo_stats.get('common_resolution', 'Unknown')}
        
        Suggest an organization strategy (folder structure, naming convention, etc.) that would work best for this collection.
        """

        try:
            response = ollama.chat(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return self._handle_ollama_error(e, "suggesting organization strategy")

    def answer_photo_question(self, question: str, context: str) -> str:
        """Answer questions about the photo collection."""
        cache_key = {"question": question, "context": context}
        cached_response = self.cache.get(cache_key)
        if cached_response:
            return cached_response

        prompt = f"""
        You are a helpful photo organization assistant. Answer the user's question about their photo collection.
        
        User question: {question}
        
        Context about the photo collection:
        {context}
        
        Provide a helpful and specific answer based on the available information.
        """

        try:
            response = ollama.chat(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
            response_content = response["message"]["content"]
            self.cache.set(cache_key, response_content)
            return response_content
        except Exception as e:
            return self._handle_ollama_error(e, "answering question")

    def generate_organization_plan(self, criteria: str, metadata_df) -> str:
        """Generate a detailed organization plan."""
        summary = self._create_metadata_summary(metadata_df)

        prompt = f"""
        Create a detailed photo organization plan based on the criteria: "{criteria}"
        
        Photo collection summary:
        {summary}
        
        Provide:
        1. Suggested folder structure
        2. File naming convention
        3. Specific steps to implement
        4. Benefits of this organization method
        """

        try:
            response = ollama.chat(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return self._handle_ollama_error(e, "generating organization plan")

    def _create_metadata_summary(self, metadata_df) -> str:
        """Create a concise summary of metadata for AI processing."""
        if metadata_df.empty:
            return "No photos found."

        summary = f"""
        Total photos: {len(metadata_df)}
        Date range: {metadata_df['created_date'].min()} to {metadata_df['created_date'].max()}
        File formats: {metadata_df['format'].value_counts().to_dict()}
        Common dimensions: {metadata_df['dimensions'].value_counts().head(5).to_dict()}
        Average file size: {metadata_df['size_bytes'].mean() / (1024*1024):.1f} MB
        """

        if "exif_Model" in metadata_df.columns:
            summary += f"\nCamera models: {metadata_df['exif_Model'].value_counts().head(3).to_dict()}"

        return summary
