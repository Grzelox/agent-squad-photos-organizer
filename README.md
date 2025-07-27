# AI Agent for Organizing Photos

A powerful local AI agent that helps you organize, analyze, and manage your photo collection using Ollama for intelligent analysis and metadata extraction. Now with advanced image analysis capabilities using gemma3:4b vision model.

## Features

- **Local AI Processing**: Uses Ollama for privacy-first AI analysis
- **AI Vision Analysis**: Analyze image content using gemma3:4b vision model
- **Image Similarity Detection**: Find similar images using perceptual hashing
- **Metadata Extraction**: Extracts comprehensive EXIF and file metadata
- **Smart Organization**: Organize photos by date, camera, or AI-suggested criteria
- **Duplicate Detection**: Find and manage duplicate photos
- **Photo Clustering**: Group similar photos using machine learning
- **Interactive CLI**: Beautiful terminal interface with rich output
- **AI Chat**: Ask questions about your photo collection
- **Statistics**: Detailed analytics about your photo collection
- **Health Checks**: System status and Ollama integration verification
- **Image Comparison**: Compare two images using AI vision

## Installation

### Prerequisites

1. **Python 3.8+**: Make sure you have Python 3.8 or higher
2. **Ollama**: Download and install from [https://ollama.ai/](https://ollama.ai/)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd agent-squad-photos-organizer
   ```

2. **Install dependencies**:
   ```bash
   uv sync  # or pip install -e .
   ```

3. **Start Ollama**:
   ```bash
   ollama serve
   ```

4. **Pull the AI vision model**:
   ```bash
   ollama pull gemma3:4b
   ```

5. **Prepare your photos**:
   ```bash
   # Copy your photos to the photos/ directory
   # or specify a custom directory with --photos-dir
   ```

## 📖 Usage

### Quick Start

```bash
# Check if everything is working
python main.py ollama-status

# Scan your photos
python main.py scan

# Get detailed statistics
python main.py stats

# Find duplicates
python main.py duplicates

# Find similar images using perceptual hashing
python main.py find-similar

# Analyze image content with AI vision
python main.py analyze-content

# Organize photos by date (dry run first)
python main.py organize --method date

# Actually organize photos
python main.py organize --method date --execute
```

### Available Commands

| Command | Description | Options | Example |
|---------|-------------|---------|---------|
| `scan` | Scan and analyze photos in directory | | `python main.py scan` |
| `stats` | Show detailed collection statistics | | `python main.py stats` |
| `duplicates` | Find duplicate photos | | `python main.py duplicates` |
| `cluster` | Group photos by similarity | `--clusters N` | `python main.py cluster --clusters 10` |
| `organize` | Organize photos by date/camera | `--method {date,camera}`, `--execute` | `python main.py organize --method date --execute` |
| `ask` | Ask AI about your photos | | `python main.py ask "How many sunset photos do I have?"` |
| `interactive` | Start interactive chat mode | | `python main.py interactive` |
| `ollama-status` | Check Ollama status and models | | `python main.py ollama-status` |
| `find-similar` | Find similar images using hashing | `--threshold FLOAT`, `--output FILE` | `python main.py find-similar --threshold 0.9` |
| `analyze-content` | Analyze image content with AI | `--output FILE` | `python main.py analyze-content` |
| `analyze-single` | Analyze a single image | `--prompt TEXT` | `python main.py analyze-single photo.jpg` |
| `compare-images` | Compare two images with AI | `--prompt TEXT` | `python main.py compare-images img1.jpg img2.jpg` |

### Organization Methods

- **By Date**: Creates `organized/by_date/YYYY/MM-Month/` structure
- **By Camera**: Creates `organized/by_camera/Camera_Model/` structure

### Custom Photos Directory

```bash
python main.py --photos-dir /path/to/your/photos scan
```

## Features

### Interactive Chat

Start an interactive session to ask questions about your photos:

```bash
python main.py interactive
```

Example questions:
- "How many photos do I have?"
- "What types of photos are in my collection?"
- "Suggest an organization strategy for my photos"
- "Which photos are the largest?"

### AI-Powered Analysis

The AI can help you with:
- **Organization suggestions** based on your collection
- **Duplicate detection** insights
- **Quality analysis** of your photos
- **Date/time pattern** recognition
- **Camera/device pattern** analysis

### AI Vision Analysis

#### Single Image Analysis

Analyze the content of a single image:

```bash
python main.py analyze-single photo.jpg
```

#### Image Comparison

Compare two images and get AI analysis:

```bash
python main.py compare-images photo1.jpg photo2.jpg
```

#### Content-Based Grouping

Group images by their content using AI vision:

```bash
python main.py analyze-content
```

This will:
- Analyze each image with the AI vision model
- Extract scene types, subjects, and tags
- Group similar images together
- Save results to a JSON file

#### Similarity Detection

Find visually similar images using perceptual hashing:

```bash
python main.py find-similar --threshold 0.85
```

This will:
- Calculate perceptual hashes for all images
- Group images with similar hashes
- Allow you to adjust similarity threshold
- Save results to a JSON file

### Advanced AI Prompts

You can customize AI analysis with custom prompts:

```bash
# Custom single image analysis
python main.py analyze-single photo.jpg --prompt "Describe the emotions in this image"

# Custom image comparison
python main.py compare-images img1.jpg img2.jpg --prompt "Which image has better composition?"
```

## Configuration

### Environment Variables

Create a `.env` file to customize settings:

```env
MODEL_NAME=gemma3:4b
PHOTOS_DIRECTORY=photos
```

### Model Requirements

The application now uses `gemma3:4b` which supports:
- **Text generation**: For answering questions and providing insights
- **Image analysis**: For analyzing photo content and comparing images
- **Multi-modal input**: Can process both text and images simultaneously

## Output Formats

### Analysis Results

Image analysis results are saved in JSON format:

```json
{
  "similar_group_1": {
    "images": [
      {"file_path": "photo1.jpg", "filename": "photo1.jpg", "hash": "..."}
    ],
    "count": 3,
    "similarity_score": 0.85
  }
}
```

### Content Analysis

Content-based grouping results:

```json
{
  "outdoor_landscape": {
    "scene_type": "outdoor",
    "main_subjects": ["mountain", "sky"],
    "images": [...],
    "tags": ["nature", "landscape", "outdoor"]
  }
}
```

