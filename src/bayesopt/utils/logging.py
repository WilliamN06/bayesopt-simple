"""Console logging utilities with colour support."""

from typing import Optional
import sys


class Colours:
    """ANSI colour codes for console output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class ConsoleLogger:
    """Console logger with colour support."""
    
    def __init__(self, verbose: int = 1, use_colours: bool = True):
        self.verbose = verbose
        self.use_colours = use_colours and self._supports_colour()
    
    def _supports_colour(self) -> bool:
        """Check if terminal supports colour."""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def _colour(self, text: str, colour: str) -> str:
        """Apply colour to text if supported."""
        if self.use_colours:
            return f"{colour}{text}{Colours.END}"
        return text
    
    def info(self, message: str, level: int = 1) -> None:
        """Log info message."""
        if self.verbose >= level:
            print(self._colour(message, Colours.BLUE))
    
    def success(self, message: str, level: int = 1) -> None:
        """Log success message."""
        if self.verbose >= level:
            print(self._colour(message, Colours.GREEN))
    
    def warning(self, message: str, level: int = 1) -> None:
        """Log warning message."""
        if self.verbose >= level:
            print(self._colour(message, Colours.YELLOW))
    
    def error(self, message: str, level: int = 0) -> None:
        """Log error message."""
        if self.verbose >= level:
            print(self._colour(message, Colours.RED))
    
    def highlight(self, message: str, level: int = 1) -> None:
        """Log highlighted message."""
        if self.verbose >= level:
            print(self._colour(message, Colours.BOLD + Colours.CYAN))
    
    def section(self, title: str, level: int = 1) -> None:
        """Log a section header."""
        if self.verbose >= level:
            width = 60
            line = "=" * width
            print(self._colour(line, Colours.BOLD))
            print(self._colour(f" {title} ".center(width), Colours.BOLD))
            print(self._colour(line, Colours.BOLD))
    
    def progress(self, current: int, total: int, message: str = "", level: int = 1) -> None:
        """Log progress update."""
        if self.verbose >= level:
            percent = current / total * 100
            bar_width = 30
            filled = int(bar_width * current / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress_str = f"[{bar}] {percent:.1f}%"
            if message:
                progress_str += f" - {message}"
            print(self._colour(progress_str, Colours.CYAN), end="\r" if current < total else "\n")
    
    def log_explanation(self, explanation: str, level: int = 1) -> None:
        """Log explanation text."""
        if self.verbose >= level:
            lines = explanation.split("\n")
            for line in lines:
                if line.startswith("  *"):
                    print(self._colour(line, Colours.GREEN))
                elif line.startswith("  "):
                    print(self._colour(line, Colours.CYAN))
                else:
                    print(self._colour(line, Colours.BLUE))
    
    def log_result(self, params: dict, value: float, level: int = 1) -> None:
        """Log result summary."""
        if self.verbose >= level:
            self.section("Result", level)
            for key, val in params.items():
                if isinstance(val, float):
                    print(self._colour(f"{key}: {val:.6f}", Colours.BOLD))
                else:
                    print(self._colour(f"{key}: {val}", Colours.BOLD))
            print(self._colour(f"Value: {value:.6f}", Colours.GREEN))