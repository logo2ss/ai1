"""Frozen-executable entry point (PyInstaller cannot run package-relative modules directly)."""

import multiprocessing

from lumen_engine.server.main import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # the MediaPipe helper process needs this when frozen
    main()
