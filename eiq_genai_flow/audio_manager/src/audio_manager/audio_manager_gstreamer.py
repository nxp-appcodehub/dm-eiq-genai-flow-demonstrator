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
GStreamer Audio Backend

Provides GStreamer-based audio capture and playback with support for
dynamic sample rate changes and hardware resampling.
"""

import logging
import threading
import numpy as np
from typing import Optional
from audio_manager.audio_manager_base import AudioManager, CaptureConfig, PlaybackConfig
import time

# Import gi and set versions BEFORE importing from gi.repository
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import Gst, GLib, GstApp  # noqa: E402

# Initialize GStreamer framework
Gst.init(None)
logger = logging.getLogger(__name__)


class AudioManagerGStreamer(AudioManager):
    """GStreamer-based audio manager with hardware acceleration support."""

    # Class-level tracking of shared GLib loop
    _shared_glib_loop = None
    _shared_glib_thread = None
    _glib_loop_refcount = 0
    _glib_loop_lock = threading.Lock()

    def __init__(
        self,
        capture_config: Optional[CaptureConfig] = None,
        playback_config: Optional[PlaybackConfig] = None,
        start_glib_loop: bool = True,
        external_glib_loop: Optional[GLib.MainLoop] = None,
    ):
        """
        Initialize GStreamer audio manager.

        Args:
            capture_config: Capture configuration
            playback_config: Playback configuration
            start_glib_loop: If True, start GLib loop automatically
                            (default: True) Set to False if your application
                            manages the GLib loop
            external_glib_loop: Use an existing GLib loop instead of
                              creating one. Useful for applications with
                              multiple GStreamer pipelines
        """
        super().__init__(capture_config, playback_config)

        # GStreamer pipeline handles
        self.capture_pipeline: Optional[Gst.Pipeline] = None
        self.playback_pipeline: Optional[Gst.Pipeline] = None

        # App source/sink elements
        self.capture_appsink: Optional[Gst.Element] = None
        self.playback_appsrc: Optional[Gst.Element] = None

        # Track current playback sample rate
        self.current_playback_sample_rate: Optional[int] = None

        # Flow control for appsrc
        self.appsrc_can_push = threading.Event()
        self.appsrc_can_push.set()

        # Event for pipeline drain detection
        self.appsrc_drained = threading.Event()

        # Event for EOS detection
        self.eos_received = threading.Event()

        # GLib loop management
        self.owns_glib_loop = False
        self.glib_loop: Optional[GLib.MainLoop] = None
        self.glib_thread: Optional[threading.Thread] = None

        # Setup GLib loop
        if external_glib_loop:
            # Use provided external loop
            logger.debug("Using external GLib main loop")
            self.glib_loop = external_glib_loop
            self.owns_glib_loop = False
        elif start_glib_loop:
            # Start shared GLib loop if needed
            self._start_glib_loop()
        else:
            # Don't start loop - application will manage it
            logger.debug("GLib loop not started - application must manage it")

        logger.debug("AudioManagerGStreamer initialized")

    def _start_glib_loop(self):
        """
        Start GLib main loop in separate thread (shared across instances).

        Uses reference counting to ensure only one loop runs, even with
        multiple AudioManagerGStreamer instances.
        """
        with AudioManagerGStreamer._glib_loop_lock:
            # Check if shared loop already exists
            if AudioManagerGStreamer._shared_glib_loop is not None:
                logger.debug("Using existing shared GLib loop")
                self.glib_loop = AudioManagerGStreamer._shared_glib_loop
                self.glib_thread = AudioManagerGStreamer._shared_glib_thread
                AudioManagerGStreamer._glib_loop_refcount += 1
                self.owns_glib_loop = False
                return

            # Create new shared loop
            logger.debug("Starting new shared GLib main loop")
            self.glib_loop = GLib.MainLoop()
            self.glib_thread = threading.Thread(target=self.glib_loop.run, name="GLib-MainLoop", daemon=True)
            self.glib_thread.start()

            # Store as shared loop
            AudioManagerGStreamer._shared_glib_loop = self.glib_loop
            AudioManagerGStreamer._shared_glib_thread = self.glib_thread
            AudioManagerGStreamer._glib_loop_refcount = 1
            self.owns_glib_loop = True

    def _stop_glib_loop(self):
        """
        Stop GLib main loop with reference counting.

        Only stops the loop when the last instance is shut down, and only
        if this instance owns or shares the loop (not external loops).
        """
        # Don't stop external loops - they're managed elsewhere
        if self.glib_loop and not self.owns_glib_loop:
            # Check if this is an external loop (not in shared pool)
            if self.glib_loop != AudioManagerGStreamer._shared_glib_loop:
                logger.debug("External GLib loop - not stopping (managed externally)")
                return

        # Don't try to stop if no loop was ever started
        if not self.glib_loop:
            logger.debug("No GLib loop to stop")
            return

        with AudioManagerGStreamer._glib_loop_lock:
            if AudioManagerGStreamer._glib_loop_refcount == 0:
                # No loop running
                logger.debug("No shared GLib loop running")
                return

            AudioManagerGStreamer._glib_loop_refcount -= 1
            refcount = AudioManagerGStreamer._glib_loop_refcount
            logger.debug(f"GLib loop refcount decreased: {refcount}")

            # Only stop if this was the last instance
            if AudioManagerGStreamer._glib_loop_refcount == 0:
                logger.info("Stopping shared GLib main loop (last instance)")
                if AudioManagerGStreamer._shared_glib_loop:
                    AudioManagerGStreamer._shared_glib_loop.quit()
                if AudioManagerGStreamer._shared_glib_thread:
                    AudioManagerGStreamer._shared_glib_thread.join(timeout=2.0)
                    if AudioManagerGStreamer._shared_glib_thread.is_alive():
                        logger.warning("GLib loop thread did not stop within timeout")

                # Clear shared references
                AudioManagerGStreamer._shared_glib_loop = None
                AudioManagerGStreamer._shared_glib_thread = None
            else:
                logger.debug(f"GLib loop still in use by {refcount} other instance(s)")

    # =========================================================================
    # Capture implementation
    # =========================================================================

    def _create_capture_pipeline(self) -> Gst.Pipeline:
        """
        Create GStreamer capture pipeline with resampling.

        Pipeline structure:
        alsasrc -> caps filter -> audioconvert -> audioresample -> caps filter -> appsink

        - alsasrc: Captures audio from ALSA device
        - caps filter: Enforces input format constraints
        - audioconvert: Converts between different audio formats
        - audioresample: Resamples to target sample rate with high quality
        - caps filter: Enforces output format for processing
        - appsink: Provides audio data to application via callbacks

        Returns:
            Configured GStreamer pipeline ready to start
        """
        blocksize = self.capture_config.blocksize_bytes

        # Capture Pipeline
        pipeline_str = (
            f"alsasrc device={self.capture_device} blocksize={blocksize} ! "
            f"queue max-size-buffers=10 leaky=downstream ! "
            f"audio/x-raw,rate={self.capture_config.sample_rate},"
            f"channels={self.capture_config.channels} ! "
            f"audioconvert ! "
            f"queue max-size-buffers=10 leaky=downstream ! "
            f"audio/x-raw,format={self.capture_config.format},"
            f"rate={self.capture_config.sample_rate},"
            f"channels={self.capture_config.channels},layout=interleaved ! "
            f"appsink name=appsink emit-signals=true sync=false max-buffers=5"
        )

        logger.debug(f"Capture pipeline: {pipeline_str}")
        pipeline = Gst.parse_launch(pipeline_str)

        # Get appsink element and connect callback for new samples
        appsink = pipeline.get_by_name("appsink")
        appsink.connect("new-sample", self._on_new_sample)
        self.capture_appsink = appsink

        return pipeline

    def _on_new_sample(self, appsink: Gst.Element) -> Gst.FlowReturn:
        """
        Handle new audio sample from capture pipeline.

        Called by GStreamer when new audio data is available from the appsink.
        This runs in the GStreamer streaming thread, so processing should be quick.

        Args:
            appsink: The appsink element that has a new sample

        Returns:
            Gst.FlowReturn.OK if successful, ERROR otherwise
        """
        try:
            # Pull sample from appsink (this removes it from the queue)
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR

            # Extract buffer containing audio data
            buffer = sample.get_buffer()
            if not buffer:
                return Gst.FlowReturn.ERROR

            # Map buffer for reading (locks the memory)
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR

            try:
                # Determine numpy dtype based on format string
                dtype_map = {"F32LE": np.float32, "S32LE": np.int32, "S16LE": np.int16}
                dtype = dtype_map.get(self.capture_config.format, np.int32)

                # Convert buffer data to numpy array
                # .copy() ensures data persists after buffer is unmapped
                audio_data = np.frombuffer(map_info.data, dtype=dtype).copy()

                # Reshape multi-channel audio to (samples, channels)
                if self.capture_config.channels > 1:
                    audio_data = audio_data.reshape(-1, self.capture_config.channels)

                # Process the captured audio (callbacks, recording, fade-in, etc.)
                self._process_captured_audio(audio_data)
            finally:
                # Always unmap buffer to release memory lock
                buffer.unmap(map_info)

            return Gst.FlowReturn.OK

        except Exception as e:
            logger.error(f"Sample processing error: {e}", exc_info=True)
            return Gst.FlowReturn.ERROR

    def start_capture(self):
        """
        Start GStreamer audio capture.

        Creates the capture pipeline and sets it to PLAYING state.
        GStreamer will start calling the new-sample callback as audio arrives.
        """
        if self.capture_running.is_set():
            return

        logger.info(
            f"Starting capture: {self.capture_device} ({self.capture_config.sample_rate}Hz, "
            f"{self.capture_config.channels}ch)"
        )

        try:
            # Create and configure pipeline
            self.capture_pipeline = self._create_capture_pipeline()

            # Reset statistics
            self.buffers_dropped = 0
            self.fade_in_done = False

            # Start audio recorder if configured
            if self.audio_capture_recorder:
                self.audio_capture_recorder.start()

            # Start the pipeline - begins audio capture
            self.capture_pipeline.set_state(Gst.State.PLAYING)
            self.capture_running.set()

        except Exception as e:
            logger.error(f"Failed to start capture: {e}", exc_info=True)
            self.capture_running.clear()

    def stop_capture(self):
        """
        Stop GStreamer audio capture.

        Sets pipeline to NULL state (stops and releases resources),
        and stops any active recording.
        """
        if not self.capture_running.is_set():
            return

        logger.info("Stopping capture")
        self.capture_running.clear()

        # Stop pipeline and release resources
        if self.capture_pipeline:
            self.capture_pipeline.set_state(Gst.State.NULL)
            self.capture_pipeline = None

        # Stop recorder
        if self.audio_capture_recorder:
            self.audio_capture_recorder.stop()

    # =========================================================================
    # Playback implementation
    # =========================================================================

    def _create_playback_pipeline(self, sample_rate: Optional[int] = None) -> Gst.Pipeline:
        """
        Create GStreamer playback pipeline.

        Uses playback_config.frame_duration_ms to calculate buffer parameters:
        - buffer-time: Total buffer size (affects stability, higher = more buffering)
        - latency-time: Minimum data before playback starts (affects latency)
        """
        sr = sample_rate or self.playback_config.sample_rate
        logger.debug(f"Creating playback pipeline: {self.playback_device} at {sr}Hz")

        # Calculate GStreamer buffer parameters from frame_duration_ms
        # GStreamer expects microseconds
        frame_duration_us = self.playback_config.frame_duration_ms * 1000

        # buffer-time: use frame duration
        buffer_time_us = frame_duration_us

        # latency-time: use 1/3 of frame duration for low latency
        # (smaller = lower latency, but more risk of underruns)
        latency_time_us = frame_duration_us // 3  # 10ms for 30ms frames

        # Playback Pipeline
        pipeline_str = (
            f"appsrc name=appsrc format=time is-live=true block=false "
            f"max-bytes=0 "
            f"emit-signals=true ! "
            f"queue max-size-buffers=10 ! "
            f"audio/x-raw,format={self.playback_config.format},rate={sr},"
            f"channels={self.playback_config.channels},layout=interleaved ! "
            f"audioconvert ! "
            f"alsasink name=alsasink device={self.playback_device} sync=true "
            f"buffer-time={buffer_time_us} latency-time={latency_time_us}"
        )

        logger.debug(f"Playback pipeline: {pipeline_str}")
        pipeline = Gst.parse_launch(pipeline_str)

        # Connect bus message handler for error/warning monitoring
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # Configure appsrc for streaming mode
        appsrc = pipeline.get_by_name("appsrc")
        appsrc.set_property("stream-type", GstApp.AppStreamType.STREAM)

        # Connect flow control signals (only when keep_device_open=True)
        if self.playback_config.keep_device_open:
            appsrc.connect("need-data", self._on_need_data)
            appsrc.connect("enough-data", self._on_enough_data)

        self.playback_appsrc = appsrc
        self.current_playback_sample_rate = sr

        # Reset flow control state for new pipeline
        if self.playback_config.keep_device_open:
            self.appsrc_can_push.set()

        # Time when playback is expected to end
        self.playback_end_ts = time.time()

        return pipeline

    def _on_need_data(self, appsrc: Gst.Element, length: int):
        """
        Callback when appsrc needs more data.

        Args:
            appsrc: The appsrc element requesting data
            length: Suggested amount of data to push (in bytes)
        """
        logger.debug(f"appsrc needs data (suggested: {length} bytes)")
        self.appsrc_can_push.set()

        # Signal that pipeline has drained (needs more data means it consumed what we gave it)
        self.appsrc_drained.set()

    def _wait_for_pipeline_drain(self, send_eos: bool = False):
        """
        Wait for GStreamer pipeline to drain its buffers.

        Args:
            send_eos: If True, send EOS and wait for EOS message (destroys pipeline state)
                     If False, wait for appsrc to drain using need-data signal (keeps pipeline alive)

        When send_eos=True:
            Sends EOS to the pipeline and waits for the EOS message on the bus,
            which indicates all audio has been played through alsasink.
            Pipeline will be in EOS state and should be destroyed.

        When send_eos=False:
            Waits for appsrc to drain using need-data signal, then waits until
            playback_end_ts (estimated completion time). Pipeline remains in PLAYING state.
        """
        if not self.playback_pipeline or not self.playback_appsrc:
            return

        logger.debug(f"_wait_for_pipeline_drain called with send_eos={send_eos}")

        if send_eos:
            # EOS mode: proper pipeline drainage with EOS signal
            logger.debug("Sending EOS to pipeline and waiting for playback to complete...")
            self.eos_received.clear()
            self.playback_appsrc.send_event(Gst.Event.new_eos())

            if not self.eos_received.wait(timeout=30.0):
                logger.warning("Timeout waiting for EOS - pipeline may not have finished properly")
            else:
                logger.debug("Pipeline finished playing all audio (EOS received)")
        else:
            # Non-EOS mode: wait for appsrc drain + wait until playback_end_ts

            # Wait for appsrc to drain first (buffer empty)
            appsrc_current_level = self.playback_appsrc.get_property("current-level-bytes")
            if appsrc_current_level > 0:
                self.appsrc_drained.clear()
                logger.debug("Waiting for appsrc to drain...")
                self.appsrc_drained.wait()
                logger.debug("appsrc drained (need-data received)")

            # Now wait until the estimated playback end time
            now = time.time()
            remaining_time = self.playback_end_ts - now

            if remaining_time > 0:
                logger.debug(f"Waiting {remaining_time:.3f}s until playback_end_ts ({self.playback_end_ts:.3f}s)")
                time.sleep(remaining_time)
            else:
                logger.debug(f"playback_end_ts already passed (was {-remaining_time:.3f}s ago)")

        logger.debug("Pipeline drain complete")

    def _on_enough_data(self, appsrc: Gst.Element):
        """
        Callback when appsrc has enough data.

        Args:
            appsrc: The appsrc element signaling enough data
        """
        logger.debug("appsrc has enough data, pausing pushes")
        self.appsrc_can_push.clear()

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message):
        """
        Handle GStreamer bus messages.

        The bus is GStreamer's message passing system for async events.
        This callback receives errors, warnings, and state changes from the pipeline.

        Args:
            bus: The message bus
            message: The message to process
        """
        if message.type == Gst.MessageType.ERROR:
            # Critical errors that stop the pipeline
            err, debug = message.parse_error()
            logger.error(f"GStreamer Error: {err}")
        elif message.type == Gst.MessageType.WARNING:
            # Non-critical warnings (degraded performance, etc.)
            warn, debug = message.parse_warning()
            logger.warning(f"GStreamer Warning: {warn}")
        elif message.type == Gst.MessageType.EOS:
            # End of stream reached
            logger.debug("EOS message received on bus")
            self.eos_received.set()

    def _push_audio_to_device(self, audio_data: np.ndarray, sample_rate: int):
        """
        Push audio to GStreamer pipeline with dynamic sample rate support.

        Handles pipeline recreation when sample rate changes and pushes
        audio buffers to the appsrc element for playback.

        Args:
            audio_data: Audio samples to play in playback_config.format (S32LE, S16LE, or F32LE)
            sample_rate: Sample rate of the audio
        """
        try:
            # GStreamer caps are fixed once negotiated, so we need a new pipeline
            if not self.playback_pipeline:
                # Create new pipeline
                self.playback_pipeline = self._create_playback_pipeline(sample_rate)
                ret = self.playback_pipeline.set_state(Gst.State.PLAYING)

                if ret == Gst.StateChangeReturn.FAILURE:
                    logger.error("Failed to set pipeline to PLAYING")
                    return

            # Channel adjustment and format conversion in base class
            playback_data = self._prepare_playback_audio(audio_data)

            # Calculate playback duration based on audio buffer size
            # This method is more reliable than querying GStreamer pipeline latency
            duration = playback_data.size / (self.current_playback_sample_rate * self.playback_config.channels)
            logger.debug(f"Pushing audio buffer: {duration:.3f}s")

            now = time.time()

            # Update playback end timestamp
            # If playback is ongoing, add to existing end time
            # Otherwise, start new playback timeline with safety margin
            PLAYBACK_SAFETY_MARGIN = 0.2  # seconds - accounts for pipeline buffering

            if self.playback_end_ts > now:
                # Playback already in progress - extend end time
                self.playback_end_ts += duration
                logger.debug(
                    f"Extended playback end time: {self.playback_end_ts:.3f}s (current: {now:.3f}s, +{duration:.3f}s)"
                )
            else:
                # Starting new playback session
                self.playback_end_ts = now + duration + PLAYBACK_SAFETY_MARGIN
                logger.debug(
                    f"New playback session end time: {self.playback_end_ts:.3f}s (current: {now:.3f}s, f"
                    f"duration: {duration:.3f}s)"
                )

            # Wait for appsrc to signal it can accept data (with timeout)
            if not self.appsrc_can_push.wait(timeout=60):
                logger.error("Timeout waiting for appsrc to accept data, dropping audio buffer")
                return

            # Wrap audio data in GStreamer buffer
            gst_buffer = Gst.Buffer.new_wrapped(playback_data.tobytes())

            # Push buffer to appsrc element
            # This queues the audio for playback through the pipeline
            ret = self.playback_appsrc.emit("push-buffer", gst_buffer)

            if ret != Gst.FlowReturn.OK:
                logger.warning(f"push-buffer returned {ret}")
                # If push failed, signal might not come, so set it anyway
                self.appsrc_can_push.set()

        except Exception as e:
            logger.error(f"Playback error: {e}", exc_info=True)

    def start_playback(self):
        """
        Start GStreamer playback thread.

        Creates a background thread that monitors the playback queue and
        sends audio to the GStreamer pipeline. Optionally pre-creates the
        pipeline if keep_device_open is enabled.
        """
        if self.playback_running.is_set():
            return

        logger.debug(f"Starting playback: {self.playback_device} ({self.playback_config.sample_rate}Hz)")

        # Pre-create pipeline if keep_device_open is enabled
        # This avoids pipeline creation overhead for each playback
        if self.playback_config.keep_device_open:
            self.playback_pipeline = self._create_playback_pipeline()
            self.playback_pipeline.set_state(Gst.State.PLAYING)

        # Start playback thread
        self.playback_running.set()
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def stop_playback(self):
        """
        Stop GStreamer playback.

        Signals the playback thread to stop, waits for it to finish,
        and cleans up the pipeline.
        """
        if not self.playback_running.is_set():
            return

        logger.info("Stopping playback")

        # Signal thread to stop
        self.playback_running.clear()

        # Wake up the thread if it's waiting
        with self.playback_queue_lock:
            self.playback_queue_cv.notify()

        # Wait for thread to finish (with timeout)
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
            if self.playback_thread.is_alive():
                logger.warning("Playback thread did not stop gracefully")

        # Stop pipeline and release resources (if not already done by loop)
        if self.playback_pipeline:
            self.playback_pipeline.set_state(Gst.State.NULL)
            self.playback_pipeline = None
            self.current_playback_sample_rate = None

        self.playback_thread = None

    def _playback_loop(self):
        """
        Playback thread loop.

        Behavior depends on keep_device_open setting:
        - If False: Sends EOS and destroys pipeline after each playback session
        - If True: Keeps pipeline alive and reuses it for next playback session
        """
        try:
            while self.playback_running.is_set():
                audio_to_play = None
                sr = self.playback_config.sample_rate

                # Get next audio chunk
                with self.playback_queue_lock:
                    if self.playback_queue:
                        # Dequeue audio and process it
                        audio_to_play, sr = self.playback_queue.popleft()
                    elif self.stream_end_signaled.is_set():
                        logger.debug("Queue empty and stream end signaled - waiting for pipeline to finish")

                        # Keep playback_active set during drain
                        self.playback_active.set()

                        # Release lock while waiting for playback to complete
                        self.playback_queue_lock.release()
                        try:
                            # Different behavior based on keep_device_open
                            if not self.playback_config.keep_device_open:
                                # Close device after playback: use EOS
                                logger.debug("Using EOS mode (keep_device_open=False)")
                                self._wait_for_pipeline_drain(send_eos=True)

                                # Destroy pipeline after EOS
                                logger.debug("Closing playback pipeline after EOS")
                                if self.playback_pipeline:
                                    self.playback_pipeline.set_state(Gst.State.NULL)
                                    self.playback_pipeline = None
                                    self.current_playback_sample_rate = None
                            else:
                                # Keep device open: just drain without EOS
                                logger.debug("Using non-EOS mode (keep_device_open=True)")
                                self._wait_for_pipeline_drain(send_eos=False)
                        finally:
                            self.playback_queue_lock.acquire()

                        # Clear playback_active AFTER playback completes
                        self.playback_active.clear()

                        logger.info("Playback complete")

                        # Clear stream end signal for next playback
                        self.stream_end_signaled.clear()

                        # Notify waiters
                        with self.playback_queue_cv:
                            self.playback_queue_cv.notify_all()

                        continue
                    else:
                        # Queue empty but stream still active - wait for more chunks
                        # Wait with timeout for new audio or stream end signal
                        self.playback_queue_cv.wait(timeout=1.0)
                        continue  # Check queue again after wait

                # Process audio chunk (outside the lock)
                if audio_to_play is not None:
                    self.playback_active.set()

                    # Create new pipeline if needed
                    if not self.playback_pipeline or sr != self.current_playback_sample_rate:
                        if self.playback_pipeline:
                            logger.info("Sample rate changed, recreating pipeline")
                            self.playback_pipeline.set_state(Gst.State.NULL)
                            self.playback_pipeline = None

                        # Create new pipeline for this playback session
                        self.playback_pipeline = self._create_playback_pipeline(sr)
                        self.playback_pipeline.set_state(Gst.State.PLAYING)

                    # Push audio to device
                    self._push_audio_to_device(audio_to_play, sr)

                    # Mark this chunk as processed
                    self.playback_active.clear()

                    # Notify waiters that we processed a chunk
                    with self.playback_queue_cv:
                        self.playback_queue_cv.notify_all()

        except Exception as e:
            logger.error(f"Playback loop error: {e}", exc_info=True)
        finally:
            # Clean up pipeline on thread exit
            if self.playback_pipeline:
                logger.debug("Playback loop stopped - cleaning up pipeline")
                self.playback_pipeline.set_state(Gst.State.NULL)
                self.playback_pipeline = None
                self.current_playback_sample_rate = None

            self.playback_active.clear()

            # Clear the stream end signal for next playback session
            with self.playback_queue_lock:
                self.stream_end_signaled.clear()
                self.playback_queue_cv.notify_all()

    def shutdown(self):
        """
        Shutdown GStreamer backend and clean up resources.

        Stops audio operations and cleans up the GLib main loop.
        Should be called before application exit to prevent resource leaks.
        """
        logger.info("Shutting down AudioManagerGStreamer")

        # Stop capture and playback (from base class)
        super().shutdown()

        # Stop GLib event loop
        self._stop_glib_loop()
