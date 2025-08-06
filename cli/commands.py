import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from modules.services.photo_service import PhotoService
from modules.services.ai_service import AIService
from modules.services.organization_service import OrganizationService
from modules.services.image_analysis_service import ImageAnalysisService
from modules.config import config


from cli.decorators import check_ollama_availability


console: Console = Console()


def get_metadata_df(ctx):
    if ctx.obj["metadata_df"] is None:
        photo_service = ctx.obj["photo_service"]
        console.print("[yellow]Extracting metadata...[/yellow]")
        ctx.obj["metadata_df"] = photo_service.extract_all_metadata()
    return ctx.obj["metadata_df"]



@click.group()
@click.option("--photos-dir", default=None, help="Directory containing photos")
@click.pass_context
def cli(ctx, photos_dir):
    ctx.ensure_object(dict)
    photos_directory = photos_dir or config.photos_directory
    ctx.obj["photos_dir"] = photos_directory
    ctx.obj["photo_service"] = PhotoService(photos_directory)
    ctx.obj["ai_service"] = AIService()
    ctx.obj["org_service"] = OrganizationService(photos_directory)
    ctx.obj["image_analysis_service"] = ImageAnalysisService(photos_directory)
    ctx.obj["metadata_df"] = None


@cli.command()
@click.pass_context
def scan(ctx):
    photo_service = ctx.obj["photo_service"]

    console.print("[bold blue]Scanning photos directory...[/bold blue]")

    if not photo_service.validate_photos_directory():
        console.print(f"[red]No photos found in {photo_service.photos_directory}[/red]")
        console.print(
            "[yellow]Make sure your photos are in the /photos directory[/yellow]"
        )
        return

    metadata_df = get_metadata_df(ctx)

    if metadata_df.empty:
        console.print("[red]No photos could be processed[/red]")
        return

    console.print(f"[green]Found {len(metadata_df)} photos[/green]")

    table = Table(title="Photo Collection Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    total_bytes = metadata_df["size_bytes"].sum()
    if total_bytes > 0:
        total_size_mb = total_bytes / (1024 * 1024)
        size_display = f"{total_size_mb:.1f} MB"
    else:
        size_display = "0.0 MB"
        console.print(
            "[yellow]Warning: Total size is 0 bytes. This might indicate an issue with file size reading.[/yellow]"
        )

    formats = metadata_df["format"].value_counts()
    date_range = (
        f"{metadata_df['created_date'].min()} to {metadata_df['created_date'].max()}"
    )

    table.add_row("Total Photos", str(len(metadata_df)))
    table.add_row("Total Size", size_display)
    table.add_row("Date Range", date_range)
    table.add_row(
        "Formats",
        ", ".join([f"{fmt}: {count}" for fmt, count in formats.head(3).items()]),
    )

    console.print(table)


@cli.command()
@click.pass_context
def stats(ctx):
    photo_service = ctx.obj["photo_service"]
    org_service = ctx.obj["org_service"]

    console.print("[bold blue]Analyzing photo collection...[/bold blue]")

    metadata_df = get_metadata_df(ctx)
    if metadata_df.empty:
        console.print("[red]No photos found to analyze[/red]")
        return

    stats = org_service.get_collection_stats(metadata_df)

    console.print("\n[bold]Collection Statistics[/bold]")

    basic_table = Table(title="Basic Information")
    basic_table.add_column("Metric", style="cyan")
    basic_table.add_column("Value", style="green")

    basic_table.add_row("Total Photos", str(stats["total_photos"]))
    basic_table.add_row("Total Size", f"{stats['total_size_mb']:.1f} MB")
    basic_table.add_row("Average File Size", f"{stats['avg_file_size_mb']:.1f} MB")
    basic_table.add_row("Date Range", stats["date_range"])

    console.print(basic_table)

    if stats.get("formats"):
        formats_table = Table(title="File Formats")
        formats_table.add_column("Format", style="cyan")
        formats_table.add_column("Count", style="green")

        for fmt, count in stats["formats"].items():
            formats_table.add_row(fmt, str(count))

        console.print(formats_table)

    if stats.get("cameras"):
        cameras_table = Table(title="Camera Models")
        cameras_table.add_column("Camera", style="cyan")
        cameras_table.add_column("Photos", style="green")

        for camera, count in list(stats["cameras"].items())[:5]:
            cameras_table.add_row(camera, str(count))

        console.print(cameras_table)


@cli.command()
@click.pass_context
def duplicates(ctx):
    """Find duplicate photos"""
    photo_service = ctx.obj["photo_service"]
    org_service = ctx.obj["org_service"]

    console.print("[bold blue]Searching for duplicate photos...[/bold blue]")

    metadata_df = get_metadata_df(ctx)
    if metadata_df.empty:
        console.print("[red]No photos found to analyze[/red]")
        return

    duplicates = org_service.find_duplicates(metadata_df)

    if not duplicates or "error" in duplicates:
        console.print("[green]No duplicate photos found![/green]")
        return

    console.print(
        f"[yellow]Found {len(duplicates)} groups of duplicate photos[/yellow]"
    )

    for i, (hash_val, group) in enumerate(duplicates.items(), 1):
        panel_content = "\n".join([f"• {file}" for file in group["files"]])
        panel_content += f"\n\nSize: {group['size_mb']:.1f} MB each"

        console.print(
            Panel(panel_content, title=f"Duplicate Group {i}", border_style="yellow")
        )


@cli.command()
@click.option("--clusters", default=5, help="Number of clusters to create")
@click.pass_context
def cluster(ctx, clusters):
    """Cluster photos into groups based on similarity"""
    photo_service = ctx.obj["photo_service"]
    org_service = ctx.obj["org_service"]

    console.print(f"[bold blue]Clustering photos into {clusters} groups...[/bold blue]")

    metadata_df = get_metadata_df(ctx)
    if metadata_df.empty:
        console.print("[red]No photos found to analyze[/red]")
        return

    cluster_info = org_service.cluster_photos(metadata_df, clusters)

    if "error" in cluster_info:
        console.print(f"[red]{cluster_info['error']}[/red]")
        return

    console.print("[green]Clustering completed![/green]\n")

    for cluster_name, info in cluster_info.items():
        panel_content = f"Photos: {info['count']}\n"
        panel_content += f"Average dimensions: {info['avg_dimensions']}\n"
        panel_content += f"Average size: {info['avg_size_mb']:.1f} MB\n\n"
        panel_content += "Sample photos:\n"
        panel_content += "\n".join([f"• {photo}" for photo in info["photos"][:5]])
        if len(info["photos"]) > 5:
            panel_content += f"\n... and {len(info['photos']) - 5} more"

        console.print(
            Panel(
                panel_content,
                title=f"{cluster_name.replace('_', ' ').title()}",
                border_style="blue",
            )
        )


@cli.command()
@click.option(
    "--method",
    type=click.Choice(["date", "camera"]),
    default="date",
    help="Organization method",
)
@click.option(
    "--execute", is_flag=True, help="Actually move files (default is dry run)"
)
@click.pass_context
def organize(ctx, method, execute):
    """Organize photos by date or camera"""
    photo_service = ctx.obj["photo_service"]
    org_service = ctx.obj["org_service"]

    action = "Organizing" if execute else "Planning organization of"
    console.print(f"[bold blue]{action} photos by {method}...[/bold blue]")

    metadata_df = get_metadata_df(ctx)
    if metadata_df.empty:
        console.print("[red]No photos found to organize[/red]")
        return

    if method == "date":
        result = org_service.organize_by_date(metadata_df, dry_run=not execute)
    elif method == "camera":
        result = org_service.organize_by_camera(metadata_df, dry_run=not execute)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    if result["dry_run"]:
        console.print("[yellow]DRY RUN - No files were moved[/yellow]")
        console.print(f"[green]Would organize {len(result['plan'])} photos[/green]")

        sample_items = list(result["plan"].items())[:5]
        for filename, info in sample_items:
            console.print(f"  {filename} → {info['target']}")

        if len(result["plan"]) > 5:
            console.print(f"  ... and {len(result['plan']) - 5} more files")

        console.print("\n[cyan]Use --execute flag to actually move the files[/cyan]")
    else:
        console.print(
            f"[green]Successfully organized {len(result['moved_files'])} photos[/green]"
        )

        if result["errors"]:
            console.print(f"[yellow]{len(result['errors'])} errors occurred:[/yellow]")
            for error in result["errors"][:3]:
                console.print(f"  • {error}")


@cli.command()
@click.argument("question")
@click.pass_context
@check_ollama_availability
def ask(ctx, question):
    """Ask the AI agent about your photos"""
    photo_service = ctx.obj["photo_service"]
    ai_service = ctx.obj["ai_service"]
    org_service = ctx.obj["org_service"]

    console.print(f"[bold blue]Analyzing: {question}[/bold blue]")

    metadata_df = get_metadata_df(ctx)
    if metadata_df.empty:
        console.print("[red]No photos found to analyze[/red]")
        return

    stats = org_service.get_collection_stats(metadata_df)
    context = f"""
    Photo collection contains {stats['total_photos']} photos.
    Total size: {stats['total_size_mb']:.1f} MB
    Date range: {stats['date_range']}
    File formats: {stats['formats']}
    """

    if stats.get("cameras"):
        context += f"\nCamera models: {stats['cameras']}"

    response = ai_service.answer_photo_question(question, context)

    console.print(Panel(response, title="AI Assistant Response", border_style="green"))


@cli.command()
@click.pass_context
@check_ollama_availability
def interactive(ctx):
    """Start interactive chat mode with the AI agent"""
    photo_service = ctx.obj["photo_service"]
    ai_service = ctx.obj["ai_service"]
    org_service = ctx.obj["org_service"]

    console.print("[bold blue]Starting interactive mode...[/bold blue]")
    console.print("[cyan]Type 'exit' or 'quit' to leave interactive mode[/cyan]\n")


    metadata_df = get_metadata_df(ctx)
    if metadata_df.empty:
        console.print("[red]No photos found to analyze[/red]")
        return

    stats = org_service.get_collection_stats(metadata_df)
    context = f"""
    Photo collection contains {stats['total_photos']} photos.
    Total size: {stats['total_size_mb']:.1f} MB
    Date range: {stats['date_range']}
    """

    console.print(f"[green]Loaded {stats['total_photos']} photos[/green]\n")

    while True:
        try:
            question = click.prompt("You", type=str)

            if question.lower() in ["exit", "quit", "bye"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            response = ai_service.answer_photo_question(question, context)

            console.print(f"[green]AI:[/green] {response}\n")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break


@cli.command()
@click.pass_context
def ollama_status(ctx):
    """Check Ollama status and available models"""
    ai_service = ctx.obj["ai_service"]

    console.print("[bold blue]Checking Ollama status...[/bold blue]")

    if ai_service.is_ollama_available():
        console.print("[green]Ollama is running[/green]")

        available_models = ai_service.get_available_models()
        if available_models:
            console.print(f"[green]Found {len(available_models)} model(s):[/green]")
            for model in available_models:
                status = "✓" if model == ai_service.model_name else "-"
                console.print(f"  {status} {model}")
        else:
            console.print("[yellow]No models found[/yellow]")
            console.print(f"[yellow]Pull a model: ollama pull {ai_service.model_name}[/yellow]")
    else:
        console.print("[red]Ollama is not available[/red]")
        console.print("[yellow]Make sure Ollama is running: ollama serve[/yellow]")
        console.print("[yellow]Install Ollama from: https://ollama.ai[/yellow]")
        console.print(f"[yellow]After installation, run: ollama pull {ai_service.model_name}[/yellow]")


@cli.command()
@click.option("--threshold", default=0.85, help="Similarity threshold (0.0-1.0)")
@click.option(
    "--output", default="similarity_analysis.json", help="Output file for results"
)
@click.pass_context
def find_similar(ctx, threshold, output):
    """Find similar images using perceptual hashing"""
    photo_service = ctx.obj["photo_service"]
    image_analysis_service = ctx.obj["image_analysis_service"]

    console.print(
        f"[bold blue]Finding similar images (threshold: {threshold})...[/bold blue]"
    )

    metadata_df = get_metadata_df(ctx)
    if metadata_df.empty:
        console.print("[red]No photos found to analyze[/red]")
        return

    console.print("[yellow]Calculating image hashes...[/yellow]")
    similar_groups = image_analysis_service.find_similar_images(metadata_df, threshold)

    if not similar_groups:
        console.print("[green]No similar image groups found![/green]")
        return

    console.print(
        f"[green]Found {len(similar_groups)} groups of similar images[/green]"
    )

    image_analysis_service.save_analysis_results(similar_groups, output)

    for group_id, group_info in similar_groups.items():
        panel_content = f"Images: {group_info['count']}\n"
        panel_content += (
            f"Similarity threshold: {group_info['similarity_score']:.2f}\n\n"
        )
        panel_content += "Files:\n"
        panel_content += "\n".join(
            [f"• {img['filename']}" for img in group_info["images"]]
        )

        console.print(
            Panel(
                panel_content, title=f"Similar Group: {group_id}", border_style="blue"
            )
        )


@cli.command()
@click.option(
    "--output", default="content_analysis.json", help="Output file for results"
)
@click.pass_context
@check_ollama_availability
def analyze_content(ctx, output):
    """Analyze image content using AI vision model"""
    photo_service = ctx.obj["photo_service"]
    ai_service = ctx.obj["ai_service"]
    image_analysis_service = ctx.obj["image_analysis_service"]

    console.print("[bold blue]Analyzing image content with AI...[/bold blue]")

    metadata_df = get_metadata_df(ctx)
    if metadata_df.empty:
        console.print("[red]No photos found to analyze[/red]")
        return

    console.print("[yellow]Processing images with AI vision...[/yellow]")
    content_groups = image_analysis_service.group_images_by_content(
        metadata_df, ai_service
    )

    if not content_groups:
        console.print("[yellow]No content groups found[/yellow]")
        return

    console.print(f"[green]Found {len(content_groups)} content-based groups[/green]")

    image_analysis_service.save_analysis_results(content_groups, output)

    for group_key, group_info in content_groups.items():
        panel_content = f"Scene type: {group_info['scene_type']}\n"
        panel_content += f"Main subjects: {', '.join(group_info['main_subjects'])}\n"
        panel_content += f"Images: {len(group_info['images'])}\n"
        panel_content += f"Tags: {', '.join(group_info['tags'][:5])}\n\n"
        panel_content += "Sample files:\n"
        panel_content += "\n".join(
            [f"• {img['filename']}" for img in group_info["images"][:3]]
        )
        if len(group_info["images"]) > 3:
            panel_content += f"\n... and {len(group_info['images']) - 3} more"

        console.print(
            Panel(
                panel_content, title=f"Content Group: {group_key}", border_style="green"
            )
        )


@cli.command()
@click.argument("image_path")
@click.option("--prompt", default=None, help="Custom analysis prompt")
@click.pass_context
@check_ollama_availability
def analyze_single(ctx, image_path, prompt):
    """Analyze a single image with AI vision"""
    ai_service = ctx.obj["ai_service"]
    image_analysis_service = ctx.obj["image_analysis_service"]

    if not image_analysis_service._is_supported_format(image_path):
        console.print(f"[red]Unsupported image format: {image_path}[/red]")
        return

    console.print(f"[bold blue]Analyzing image: {image_path}[/bold blue]")

    if prompt is None:
        prompt = """
        Analyze this image and provide detailed information about:
        1. What you see in the image (objects, people, animals, etc.)
        2. The scene type (indoor/outdoor, landscape, portrait, etc.)
        3. Colors and lighting
        4. Image composition and quality
        5. Any text or signs visible
        6. Suggested tags for organization
        
        Provide a comprehensive analysis in a clear, structured format.
        """

    analysis = image_analysis_service.analyze_image_content(image_path, ai_service)

    if "error" in analysis:
        console.print(f"[red]{analysis['error']}[/red]")
        return

    console.print(Panel(str(analysis), title="AI Image Analysis", border_style="green"))


@cli.command()
@click.argument("image1")
@click.argument("image2")
@click.option("--prompt", default=None, help="Custom comparison prompt")
@click.pass_context
@check_ollama_availability
def compare_images(ctx, image1, image2, prompt):
    """Compare two images using AI vision"""
    ai_service = ctx.obj["ai_service"]
    image_analysis_service = ctx.obj["image_analysis_service"]

    for img_path in [image1, image2]:
        if not image_analysis_service._is_supported_format(img_path):
            console.print(f"[red]Unsupported image format: {img_path}[/red]")
            return

    console.print(f"[bold blue]Comparing images: {image1} vs {image2}[/bold blue]")

    img1_base64 = image_analysis_service.encode_image_to_base64(image1)
    img2_base64 = image_analysis_service.encode_image_to_base64(image2)

    if not img1_base64 or not img2_base64:
        console.print("[red]Failed to encode one or both images[/red]")
        return

    comparison = ai_service.compare_images(img1_base64, img2_base64, prompt)

    console.print(Panel(comparison, title="AI Image Comparison", border_style="blue"))


if __name__ == "__main__":
    cli()
