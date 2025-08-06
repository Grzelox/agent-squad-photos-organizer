
import click
from functools import wraps
from rich.console import Console

console = Console()

def check_ollama_availability(func):
    @wraps(func)
    def wrapper(ctx, *args, **kwargs):
        ai_service = ctx.obj["ai_service"]
        if not ai_service.is_ollama_available():
            console.print("[red]Ollama is not available[/red]")
            console.print("[yellow]Make sure Ollama is running: ollama serve[/yellow]")
            console.print("[yellow]Install Ollama from: https://ollama.ai[/yellow]")
            console.print(f"[yellow]After installation, run: ollama pull {ai_service.model_name}[/yellow]")
            return
        return func(ctx, *args, **kwargs)
    return wrapper
