# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

"""
Audio Manager Factory

Provides factory function to create the appropriate AudioManager implementation
based on available backends and user preference.
"""

import logging
from typing import Optional, Type
from audio_manager.audio_manager_base import CaptureConfig, PlaybackConfig, AudioManager

logger = logging.getLogger(__name__)


# Backend configuration: maps backend name to (module_path, class_name)
_BACKEND_REGISTRY = {
    "gstreamer": ("audio_manager.audio_manager_gstreamer", "AudioManagerGStreamer"),
    "alsa": ("audio_manager.audio_manager_alsa", "AudioManagerALSA"),
}


def _load_backend(backend_name: str) -> Optional[Type[AudioManager]]:
    """
    Dynamically load a backend class.

    Args:
        backend_name: Name of backend ("gstreamer" or "alsa")

    Returns:
        Backend class or None if not available
    """
    if backend_name not in _BACKEND_REGISTRY:
        return None

    module_path, class_name = _BACKEND_REGISTRY[backend_name]

    try:
        module = __import__(module_path, fromlist=[class_name])
        backend_class = getattr(module, class_name)
        logger.debug(f"Successfully loaded backend: {backend_name}")
        return backend_class
    except ImportError as e:
        logger.debug(f"Backend '{backend_name}' not available: {e}")
        return None


def create_audio_manager(
    backend: str = "auto",
    capture_config: Optional[CaptureConfig] = None,
    playback_config: Optional[PlaybackConfig] = None,
    start_glib_loop: bool = True,
    external_glib_loop: Optional[object] = None,  # GLib.MainLoop
) -> AudioManager:
    """
    Create audio manager with specified backend.

    Args:
        backend: "auto", "alsa", or "gstreamer"
        capture_config: Capture configuration
        playback_config: Playback configuration
        start_glib_loop: Start GLib loop automatically (GStreamer only)
        external_glib_loop: Use existing GLib loop (GStreamer only)

    Returns:
        AudioManager instance (GStreamer or ALSA implementation)

    Raises:
        ImportError: If requested backend is not available
        ValueError: If backend parameter is invalid
    """
    backend = backend.lower()

    # Handle auto-selection (try GStreamer, fallback to ALSA)
    if backend == "auto":
        # Try GStreamer first
        backend_class = _load_backend("gstreamer")
        if backend_class:
            logger.info("Auto-selected GStreamer audio backend")
            return backend_class(
                capture_config=capture_config,
                playback_config=playback_config,
                start_glib_loop=start_glib_loop,
                external_glib_loop=external_glib_loop,
            )

        # Fallback to ALSA
        logger.warning("GStreamer not available, falling back to ALSA")
        backend = "alsa"

    # Load specified backend
    if backend in _BACKEND_REGISTRY:
        backend_class = _load_backend(backend)
        if backend_class:
            logger.debug(f"Using {backend.upper()} audio backend")
            if backend == "gstreamer":
                return backend_class(
                    capture_config=capture_config,
                    playback_config=playback_config,
                    start_glib_loop=start_glib_loop,
                    external_glib_loop=external_glib_loop,
                )
            else:
                return backend_class(
                    capture_config=capture_config,
                    playback_config=playback_config,
                )
        else:
            raise ImportError(f"{backend.upper()} backend not available. Please install required dependencies.")
    else:
        # Invalid backend name
        valid_backends = ", ".join(["'auto'"] + [f"'{k}'" for k in _BACKEND_REGISTRY.keys()])
        raise ValueError(f"Invalid backend: '{backend}'. Must be one of: {valid_backends}")


def get_available_backends() -> dict:
    """
    Check which audio backends are available.

    Returns:
        Dictionary with backend availability status
    """
    backends = {}
    for backend_name in _BACKEND_REGISTRY.keys():
        backends[backend_name] = _load_backend(backend_name) is not None
    return backends


def print_backend_info():
    """Print information about available audio backends."""
    backends = get_available_backends()

    print("Audio Backend Availability:")
    print("-" * 40)
    for backend, available in backends.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {backend.upper():<12} {status}")
    print("-" * 40)

    if not any(backends.values()):
        print("WARNING: No audio backends available!")
    elif backends.get("gstreamer"):
        print("Recommended: GStreamer (default for 'auto')")
    elif backends.get("alsa"):
        print("Available: ALSA only")


if __name__ == "__main__":
    # Print backend info when run directly
    print_backend_info()
