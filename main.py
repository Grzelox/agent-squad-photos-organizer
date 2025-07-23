import sys
from pathlib import Path
from cli.commands import cli

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    cli()
