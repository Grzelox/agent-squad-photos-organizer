import shutil
from pathlib import Path
from typing import Dict, List
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class OrganizationService:
    """Service for organizing and analyzing photo collections."""

    def __init__(self, photos_directory: str = "photos"):
        self.photos_directory = Path(photos_directory)
        self.organized_directory = self.photos_directory / "organized"

    def find_similar_photos(
        self, metadata_df: pd.DataFrame, similarity_threshold: float = 0.8
    ) -> List[Dict]:
        """Find similar photos based on metadata."""
        if metadata_df.empty:
            return []

        numeric_cols = ["width", "height", "size_bytes", "aspect_ratio"]
        available_cols = [col for col in numeric_cols if col in metadata_df.columns]

        if len(available_cols) < 2:
            return []

        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(metadata_df[available_cols])

            similarity_matrix = cosine_similarity(scaled_data)

            similar_pairs = []
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    similarity = similarity_matrix[i][j]
                    if similarity > similarity_threshold:
                        similar_pairs.append(
                            {
                                "photo1": metadata_df.iloc[i]["filename"],
                                "photo2": metadata_df.iloc[j]["filename"],
                                "similarity": round(similarity, 3),
                                "path1": metadata_df.iloc[i]["filepath"],
                                "path2": metadata_df.iloc[j]["filepath"],
                            }
                        )

            return similar_pairs

        except Exception as e:
            print(f"Error finding similar photos: {e}")
            return []

    def cluster_photos(self, metadata_df: pd.DataFrame, n_clusters: int = 5) -> Dict:
        """Cluster photos into groups based on metadata."""
        if metadata_df.empty:
            return {}

        feature_cols = ["width", "height", "size_bytes", "aspect_ratio"]
        available_cols = [col for col in feature_cols if col in metadata_df.columns]

        if len(available_cols) < 2:
            return {"error": "Insufficient metadata for clustering"}

        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(metadata_df[available_cols])

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)

            metadata_df = metadata_df.copy()
            metadata_df["cluster"] = clusters

            cluster_info = {}
            for cluster_id in range(n_clusters):
                cluster_photos = metadata_df[metadata_df["cluster"] == cluster_id]
                cluster_info[f"cluster_{cluster_id}"] = {
                    "count": len(cluster_photos),
                    "photos": cluster_photos["filename"].tolist(),
                    "avg_dimensions": f"{cluster_photos['width'].mean():.0f}x{cluster_photos['height'].mean():.0f}",
                    "avg_size_mb": cluster_photos["size_bytes"].mean() / (1024 * 1024),
                }

            return cluster_info

        except Exception as e:
            return {"error": f"Error clustering photos: {e}"}

    def find_duplicates(self, metadata_df: pd.DataFrame) -> Dict:
        """Find potential duplicate photos based on file hash."""
        if metadata_df.empty:
            return {}

        try:
            duplicates = (
                metadata_df[metadata_df.duplicated(subset=["file_hash"], keep=False)]
                .groupby("file_hash")
                .agg({"filename": list, "filepath": list, "size_bytes": "first"})
                .to_dict("index")
            )

            duplicate_groups = {}
            for hash_val, group_data in duplicates.items():
                duplicate_groups[hash_val] = {
                    "files": group_data["filename"],
                    "paths": group_data["filepath"],
                    "size_mb": group_data["size_bytes"] / (1024 * 1024),
                    "count": len(group_data["filename"]),
                }

            return duplicate_groups

        except Exception as e:
            return {"error": f"Error finding duplicates: {e}"}

    def organize_by_date(self, metadata_df: pd.DataFrame, dry_run: bool = True) -> Dict:
        """Organize photos by date into year/month folders."""
        if metadata_df.empty:
            return {"error": "No photos to organize"}

        organization_plan = {}
        moved_files = []
        errors = []

        try:
            for _, photo in metadata_df.iterrows():
                photo_date = pd.to_datetime(photo["created_date"])
                year = photo_date.year
                month = f"{photo_date.month:02d}-{photo_date.strftime('%B')}"

                target_dir = self.organized_directory / "by_date" / str(year) / month
                target_path = target_dir / photo["filename"]

                organization_plan[photo["filename"]] = {
                    "source": photo["filepath"],
                    "target": str(target_path),
                    "date": photo_date.strftime("%Y-%m-%d"),
                }

                if not dry_run:
                    try:
                        target_dir.mkdir(parents=True, exist_ok=True)

                        shutil.move(photo["filepath"], target_path)
                        moved_files.append(photo["filename"])

                    except Exception as e:
                        errors.append(f"Error moving {photo['filename']}: {e}")

            return {
                "plan": organization_plan,
                "moved_files": moved_files,
                "errors": errors,
                "dry_run": dry_run,
            }

        except Exception as e:
            return {"error": f"Error organizing by date: {e}"}

    def organize_by_camera(
        self, metadata_df: pd.DataFrame, dry_run: bool = True
    ) -> Dict:
        """Organize photos by camera model."""
        if metadata_df.empty:
            return {"error": "No photos to organize"}

        if "exif_Model" not in metadata_df.columns:
            return {"error": "No camera information available in metadata"}

        organization_plan = {}
        moved_files = []
        errors = []

        try:
            for _, photo in metadata_df.iterrows():
                camera_model = photo.get("exif_Model", "Unknown_Camera")
                camera_folder = self._clean_folder_name(camera_model)

                target_dir = self.organized_directory / "by_camera" / camera_folder
                target_path = target_dir / photo["filename"]

                organization_plan[photo["filename"]] = {
                    "source": photo["filepath"],
                    "target": str(target_path),
                    "camera": camera_model,
                }

                if not dry_run:
                    try:
                        target_dir.mkdir(parents=True, exist_ok=True)

                        shutil.move(photo["filepath"], target_path)
                        moved_files.append(photo["filename"])

                    except Exception as e:
                        errors.append(f"Error moving {photo['filename']}: {e}")

            return {
                "plan": organization_plan,
                "moved_files": moved_files,
                "errors": errors,
                "dry_run": dry_run,
            }

        except Exception as e:
            return {"error": f"Error organizing by camera: {e}"}

    def get_collection_stats(self, metadata_df: pd.DataFrame) -> Dict:
        """Get comprehensive statistics about the photo collection."""
        if metadata_df.empty:
            return {}

        stats = {
            "total_photos": len(metadata_df),
            "total_size_mb": metadata_df["size_bytes"].sum() / (1024 * 1024),
            "avg_file_size_mb": metadata_df["size_bytes"].mean() / (1024 * 1024),
            "date_range": f"{metadata_df['created_date'].min()} to {metadata_df['created_date'].max()}",
            "formats": metadata_df["format"].value_counts().to_dict(),
            "common_resolutions": metadata_df["dimensions"]
            .value_counts()
            .head(5)
            .to_dict(),
            "aspect_ratios": metadata_df["aspect_ratio"]
            .value_counts()
            .head(5)
            .to_dict(),
        }

        if "exif_Model" in metadata_df.columns:
            stats["cameras"] = metadata_df["exif_Model"].value_counts().to_dict()

        duplicates = self.find_duplicates(metadata_df)
        if "error" not in duplicates:
            stats["duplicate_groups"] = len(duplicates)
            stats["duplicate_files"] = sum(
                group["count"] for group in duplicates.values()
            )

        return stats

    def _clean_folder_name(self, name: str) -> str:
        """Clean a string to be used as a folder name."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, "_")

        name = "_".join(name.split())
        return name[:50]  # Limit folder name length
