"""Prothynesis Runtime Application.

This is the main entry point that coordinates:
- First-run setup
- Hardware detection
- GUI launch
- Settings persistence
"""

from __future__ import annotations

import sys
import signal
from pathlib import Path

from src.runtime.settings import SettingsManager
from src.runtime.startup import run_startup


def main():
    """Main entry point."""
    settings_path = Path.home() / ".local" / "Prothynesis" / "settings.yaml"
    
    # Run startup sequence (first-run detection, etc.)
    startup_result = run_startup(settings_path)
    
    if not startup_result.get("ready", False):
        print("Runtime startup failed.")
        return 1
    
    # Launch GUI
    from src.runtime.gui import RuntimeGUI
    gui = RuntimeGUI(settings_path)
    
    def signal_handler(signum, frame):
        gui.runtime.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    return gui.run()


if __name__ == "__main__":
    sys.exit(main())
