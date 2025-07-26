import base64
import cv2
import numpy as np
import imagehash
from PIL import Image
from typing import Dict, Optional
import pandas as pd
from pathlib import Path
import json


class ImageAnalysisService:
    """Service for analyzing images and finding similarities."""

    def __init__(self, photos_directory: str):
        self.photos_directory = photos_directory
        self.supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode an image to base64 string for model input."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                return encoded_string
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def get_image_hash(self, image_path: str) -> Optional[str]:
        """Get perceptual hash of an image for similarity detection."""
        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((64, 64))
                hash_value = imagehash.average_hash(img)
                return str(hash_value)
        except Exception as e:
            print(f"Error calculating hash for {image_path}: {e}")
            return None

    def calculate_image_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two image hashes."""
        try:
            hash_obj1 = imagehash.hex_to_hash(hash1)
            hash_obj2 = imagehash.hex_to_hash(hash2)
            distance = hash_obj1 - hash_obj2
            max_distance = 64  # Maximum possible hamming distance for 64-bit hash
            similarity = 1 - (distance / max_distance)
            return similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def extract_image_features(self, image_path: str) -> Dict:
        """Extract various features from an image."""
        features = {}
        try:
            img = cv2.imread(image_path)
            if img is None:
                return features

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            features["width"] = img.shape[1]
            features["height"] = img.shape[0]
            features["aspect_ratio"] = img.shape[1] / img.shape[0]
            features["total_pixels"] = img.shape[0] * img.shape[1]

            features["mean_color"] = np.mean(img_rgb, axis=(0, 1)).tolist()
            features["std_color"] = np.std(img_rgb, axis=(0, 1)).tolist()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features["brightness"] = np.mean(gray)
            features["brightness_std"] = np.std(gray)

            edges = cv2.Canny(gray, 50, 150)
            features["edge_density"] = np.sum(edges > 0) / features["total_pixels"]

            pixels = img_rgb.reshape(-1, 3)
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            features["dominant_colors"] = kmeans.cluster_centers_.tolist()

        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")

        return features

    def find_similar_images(
        self, metadata_df: pd.DataFrame, similarity_threshold: float = 0.85
    ) -> Dict:
        """Find groups of similar images based on perceptual hashing."""
        similar_groups = {}
        processed_hashes = set()

        hash_data = []
        for _, row in metadata_df.iterrows():
            image_path = row["filepath"]
            if not self._is_supported_format(image_path):
                continue

            image_hash = self.get_image_hash(image_path)
            if image_hash:
                hash_data.append(
                    {
                        "filepath": image_path,
                        "filename": row["filename"],
                        "hash": image_hash,
                    }
                )

        for i, img1 in enumerate(hash_data):
            if img1["hash"] in processed_hashes:
                continue

            similar_images = [img1]
            processed_hashes.add(img1["hash"])

            for img2 in hash_data[i + 1 :]:
                if img2["hash"] in processed_hashes:
                    continue

                similarity = self.calculate_image_similarity(img1["hash"], img2["hash"])
                if similarity >= similarity_threshold:
                    similar_images.append(img2)
                    processed_hashes.add(img2["hash"])

            if len(similar_images) > 1:
                group_id = f"similar_group_{len(similar_groups) + 1}"
                similar_groups[group_id] = {
                    "images": similar_images,
                    "count": len(similar_images),
                    "similarity_score": similarity_threshold,
                }

        return similar_groups

    def analyze_image_content(self, image_path: str, ai_service) -> Dict:
        """Analyze image content using the AI model."""
        try:
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                return {"error": "Failed to encode image"}

            prompt = """
            Analyze this image and provide the following information in JSON format:
            1. Main subjects/objects in the image
            2. Scene type (indoor/outdoor, landscape, portrait, etc.)
            3. Colors and lighting
            4. Image quality assessment
            5. Suggested tags for organization
            
            Respond with a JSON object containing these fields.
            """

            response = ai_service.analyze_image_with_text(prompt, base64_image)

            try:
                import json

                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                return {"analysis": response}

        except Exception as e:
            return {"error": f"Error analyzing image: {e}"}

    def group_images_by_content(self, metadata_df: pd.DataFrame, ai_service) -> Dict:
        """Group images by content analysis using AI."""
        content_groups = {}

        for _, row in metadata_df.iterrows():
            image_path = row["filepath"]
            if not self._is_supported_format(image_path):
                continue

            analysis = self.analyze_image_content(image_path, ai_service)

            if "error" in analysis:
                continue

            tags = analysis.get("suggested_tags", [])
            scene_type = analysis.get("scene_type", "unknown")
            main_subjects = analysis.get("main_subjects", [])

            group_key = f"{scene_type}_{'_'.join(main_subjects[:2])}"
            group_key = group_key.replace(" ", "_").lower()

            if group_key not in content_groups:
                content_groups[group_key] = {
                    "scene_type": scene_type,
                    "main_subjects": main_subjects,
                    "images": [],
                    "tags": tags,
                }

            content_groups[group_key]["images"].append(
                {
                    "filepath": image_path,
                    "filename": row["filename"],
                    "analysis": analysis,
                }
            )

        content_groups = {
            k: v for k, v in content_groups.items() if len(v["images"]) > 1
        }

        return content_groups

    def _is_supported_format(self, filepath: str) -> bool:
        """Check if file format is supported for image analysis."""
        return Path(filepath).suffix.lower() in self.supported_formats

    def save_analysis_results(self, results: Dict, output_path: str):
        """Save analysis results to a JSON file."""
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Analysis results saved to {output_path}")
        except Exception as e:
            print(f"Error saving analysis results: {e}")

    def load_analysis_results(self, input_path: str) -> Dict:
        """Load analysis results from a JSON file."""
        try:
            with open(input_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading analysis results: {e}")
            return {}
