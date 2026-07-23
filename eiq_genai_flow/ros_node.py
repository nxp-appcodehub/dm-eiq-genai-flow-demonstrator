# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import sys
import json
import queue
import rclpy
import threading
import traceback
import numpy as np
from vit.vit import VIT
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from eiq_genai_flow.__main__ import eIQGenAIFlow, Config
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from shared_utils.utils import get_default_capture_device, get_default_playback_device


class StdinQueueReader:
    """
    Custom stdin replacement that reads from a queue.
    Used to intercept stdin calls from the GenAI Flow pipeline and redirect
    them to a thread-safe queue controlled by ROS 2 topics and services.
    """
    def __init__(self, input_queue):
        self.queue = input_queue
        self._is_ros_stdin = True

    def readline(self):
        """Read a line from the queue (blocking with timeout)."""
        while True:
            try:
                line = self.queue.get(timeout=0.1)
                return line if line.endswith('\n') else line + '\n'
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[StdinQueueReader] Exception in readline: {e}", flush=True)
                return '\n'

    def read(self, size=-1):
        """Read from queue."""
        return self.readline()

    def isatty(self):
        return False

    def fileno(self):
        # KeyboardAdapter needs a real terminal fd for termios/tty/select.
        # Open /dev/tty so it refers to the controlling terminal instead of
        # the replaced stdin queue object.
        if not hasattr(self, '_tty_fd'):
            import os
            self._tty_fd = os.open('/dev/tty', os.O_RDWR)
        return self._tty_fd


class GenAIFlowNode(Node):
    """
    ROS 2 Node wrapper for eIQ GenAI Flow.

    Publishes events from the GenAI Flow pipeline:
    - /genai/wakeword:          Wakeword detection events
    - /genai/vad/event:         VAD speech start and end events
    - /genai/voice_id/event:    All Voice ID events as a JSON string:
                                  { "type": str, "speaker": str, "confidence": float, "status": str }
    - /genai/stt/transcription: Speech-to-text final transcription
    - /genai/rag/category:      RAG query category (CENSORED, INTENT, REJECTED, AMBIGUOUS, ACCEPTED)
    - /genai/rag/answer:        RAG answer (if available)
    - /genai/llm/token:         LLM token stream events
    - /genai/llm/response:      LLM complete response
    - /genai/tts/event:         TTS events (start/end of synthesis and playback)

    Provides services:
    - /genai/trigger_listening: Trigger STT listening (replaces Enter key in kasr mode)

    Subscribes to:
    - /genai/text_input:        Send text queries to the system (keyb mode)
    """

    def __init__(self, node_name: str = "eiq_genai_flow_node"):
        super().__init__(node_name)

        # Declare node parameters with default values.
        self.declare_parameter('input_mode', 'vasr')
        self.declare_parameter('capture_device', get_default_capture_device())
        self.declare_parameter('llm_model', 'danube-500M-q8')
        self.declare_parameter('output_mode', 'tts')
        self.declare_parameter('playback_device', get_default_playback_device())
        self.declare_parameter('stt_model', 'moonshine-base')
        self.declare_parameter('system_prompt', 'Helpful assistant.')
        self.declare_parameter('use_rag', False)
        self.declare_parameter('use_neutron', True)
        self.declare_parameter('use_voice_id', False)
        self.declare_parameter('continuous', False)
        self.declare_parameter('verbose', True)

        # Create publishers
        self.wakeword_pub = self.create_publisher(String, '/genai/wakeword', 10)
        self.vad_pub = self.create_publisher(String, '/genai/vad/event', 10)
        self.voice_id_event_pub = self.create_publisher(String, '/genai/voice_id/event', 10)
        self.stt_trans_pub = self.create_publisher(String, '/genai/stt/transcription', 10)
        self.rag_cat_pub = self.create_publisher(String, '/genai/rag/category', 10)
        self.rag_answer_pub = self.create_publisher(String, '/genai/rag/answer', 10)
        self.llm_token_pub = self.create_publisher(String, '/genai/llm/token', 10)
        self.llm_response_pub = self.create_publisher(String, '/genai/llm/response', 10)
        self.tts_event_pub = self.create_publisher(String, '/genai/tts/event', 10)

        # Create service and subscription
        self.trigger_listening_srv = self.create_service(
            Trigger,
            '/genai/trigger_listening',
            self._trigger_listening_callback
        )
        self.text_input_sub = self.create_subscription(
            String,
            '/genai/text_input',
            self._text_input_callback,
            10
        )

        # Thread-safe queues
        self.event_queue = queue.Queue()
        self.stdin_queue = queue.Queue()
        self.input_mode = self.get_parameter('input_mode').value

        # Replace sys.stdin with queue-based reader
        self.original_stdin = sys.stdin
        sys.stdin = StdinQueueReader(self.stdin_queue)

        self._init_genai_flow()

        # Timer to drain and publish the event queue at 100Hz
        self.create_timer(0.01, self._process_event_queue)

    def _init_genai_flow(self):
        """Initialize the eIQ GenAI Flow pipeline and start the run loop."""
        self.get_logger().info('Initializing eIQ GenAI Flow...')
        config = Config()

        try:
            llm_model = self.get_parameter('llm_model').value
            self.genai_flow = eIQGenAIFlow(
                config=config,
                input_mode=self.get_parameter('input_mode').value,
                capture_device=self.get_parameter('capture_device').value,
                wake_word_model=VIT.get_default_model_path(),
                gui_config_class=None,
                llm_model=llm_model if llm_model != 'no_llm' else None,
                output_mode=self.get_parameter('output_mode').value,
                playback_device=self.get_parameter('playback_device').value,
                stt_model=self.get_parameter('stt_model').value,
                system_prompt=self.get_parameter('system_prompt').value,
                use_rag=self.get_parameter('use_rag').value,
                use_neutron=self.get_parameter('use_neutron').value,
                continuous=self.get_parameter('continuous').value,
                benchmark=False,
                use_voice_id=self.get_parameter('use_voice_id').value,
                verbose=self.get_parameter('verbose').value
            )

            self._patch_keyboard_adapter()
            self._register_callbacks()

            self.get_logger().info('eIQ GenAI Flow initialized successfully')

            self.flow_thread = threading.Thread(
                target=self._run_genai_flow_loop,
                daemon=True
            )
            self.flow_thread.start()
            self.get_logger().info('eIQ GenAI Flow run loop started')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize GenAI Flow: {e}')
            self.get_logger().error(traceback.format_exc())
            sys.stdin = self.original_stdin
            raise

    def _patch_keyboard_adapter(self):
        """
        Patch KeyboardAdapter._worker_loop at class level to use the stdin queue
        instead of sys.stdin.fileno() / select / tty, which are unavailable when
        sys.stdin is replaced with StdinQueueReader.
        """
        try:
            from eiq_genai_flow.adapters.keyboard import KeyboardAdapter
            from eiq_genai_flow.adapters.event_manager import EventType

            stdin_queue = self.stdin_queue

            def _worker_loop_patched(self_adapter):
                while not self_adapter._stop_event.is_set():
                    try:
                        user_input = stdin_queue.get(timeout=0.1)
                        user_input = user_input.rstrip('\n').strip()

                        if not user_input:
                            # Empty input = Enter pressed (kasr wake trigger)
                            self_adapter.publish(EventType.KEYBOARD_WAKE)
                        else:
                            # Text input: publish wake + text + end-of-input
                            self_adapter.publish(EventType.KEYBOARD_WAKE)
                            self_adapter.publish(EventType.INPUT_TEXT, data=user_input)
                            self_adapter.publish(EventType.END_OF_INPUT)

                        # Break after one input; adapter is restarted by the
                        # context manager on the next run() iteration
                        break

                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f'[KeyboardAdapter patched] Exception: {e}')
                        print(traceback.format_exc())
                        break

            KeyboardAdapter._worker_loop = _worker_loop_patched
            self.get_logger().info('KeyboardAdapter._worker_loop patched successfully')

        except ImportError as e:
            self.get_logger().error(f'Could not import KeyboardAdapter: {e}')
            self.get_logger().error(traceback.format_exc())

    def _register_callbacks(self):
        """
        Register callbacks on GenAI Flow components to capture pipeline events
        and forward them to the ROS 2 event queue for publishing.
        """
        from eiq_genai_flow.adapters.event_manager import EventType, Event

        # VIT wakeword
        if self.genai_flow.vit:
            def vit_wake_callback(event: Event):
                self.event_queue.put(('wakeword', event.data))

            self.genai_flow.event_manager.subscribe(EventType.VIT_WAKE, vit_wake_callback)
            self.get_logger().info('VIT callback registered')

        # VAD and STT
        if self.genai_flow.stt:
            def vad_speech_start_callback(event: Event):
                self.event_queue.put(('vad_state', True))

            def vad_speech_end_callback(event: Event):
                self.event_queue.put(('vad_state', False))

            def stt_transcription_callback(event: Event):
                self.event_queue.put(('stt_transcription', event.data))

            self.genai_flow.event_manager.subscribe(EventType.VAD_SPEECH_START, vad_speech_start_callback)
            self.genai_flow.event_manager.subscribe(EventType.VAD_SPEECH_END, vad_speech_end_callback)
            self.genai_flow.event_manager.subscribe(EventType.INPUT_TEXT, stt_transcription_callback)
            self.get_logger().info('VAD and STT callbacks registered')

        # Voice ID
        if self.genai_flow.voice_id:
            self._register_voice_id_callbacks()

        # RAG query classifier
        if self.genai_flow.query_classifier:
            original_classify = self.genai_flow.query_classifier.__call__

            def rag_callback(self_classifier, *args, **kwargs):
                result = original_classify(**kwargs)
                if result:
                    query_category, chunk_list, _, metadata_list = result
                    self.event_queue.put(('rag_category', query_category))
                    if query_category == "ACCEPTED" and chunk_list:
                        self.event_queue.put(('rag_answer', chunk_list[0]))
                return result

            self.genai_flow.query_classifier.__class__.__call__ = rag_callback
            self.get_logger().info('RAG callback registered')

            original_send_domain_response = self.genai_flow.send_domain_response

            def send_domain_response_patched(response_text):
                self.event_queue.put(('rag_answer', response_text))
                original_send_domain_response(response_text)

            self.genai_flow.send_domain_response = send_domain_response_patched
            self.get_logger().info('send_domain_response patched for CENSORED/REJECTED/AMBIGUOUS answers')

        # LLM token stream
        if self.genai_flow.llm:
            llm_instance = self.genai_flow.llm
            node_ref = self

            class LLMCallbackWrapper:
                """
                Pure-Python wrapper around the LLM instance.
                Intercepts __call__ to forward tokens to the ROS 2 event queue
                without mutating the (immutable C-extension) class of the original.
                All other attribute accesses are forwarded transparently.
                """
                def __init__(self, wrapped):
                    object.__setattr__(self, '_wrapped', wrapped)

                def __call__(self, *args, **kwargs):
                    full_response = ""
                    try:
                        for token in object.__getattribute__(self, '_wrapped')(*args, **kwargs):
                            node_ref.event_queue.put(('llm_token', token))
                            if token:
                                full_response += token
                            yield token
                    except Exception as e:
                        node_ref.get_logger().error(f'[LLM] Error during token generation: {e}')
                        node_ref.get_logger().error(traceback.format_exc())
                    finally:
                        if full_response:
                            node_ref.event_queue.put(('llm_response', full_response))
                        else:
                            node_ref.get_logger().warning('[LLM] No response content was generated')

                def __getattr__(self, name):
                    return getattr(object.__getattribute__(self, '_wrapped'), name)

                def __setattr__(self, name, value):
                    setattr(object.__getattribute__(self, '_wrapped'), name, value)

            self.genai_flow.llm = LLMCallbackWrapper(llm_instance)
            self.get_logger().info('LLM callback registered')

        # TTS synthesis and playback
        if self.genai_flow.tts:
            tts_original_generate = self.genai_flow.tts._generate_and_queue_audio

            def tts_synthesis_callback(*args, **kwargs):
                self.event_queue.put(('tts_synthesis', 'TTS synthesis started'))
                tts_original_generate(*args, **kwargs)
                self.event_queue.put(('tts_synthesis', 'TTS synthesis completed'))

            self.genai_flow.tts._generate_and_queue_audio = tts_synthesis_callback

            self._tts_playback_state = self.genai_flow.tts.audio_manager.is_playback_complete()

            def monitor_tts_playback():
                current = self.genai_flow.tts.audio_manager.is_playback_complete()
                if current != self._tts_playback_state:
                    self.event_queue.put(('tts_playback', current))
                    self._tts_playback_state = current

            self.create_timer(0.05, monitor_tts_playback)
            self.get_logger().info('TTS callbacks registered')

    def _voice_id_event(self, event_type, speaker='unknown', confidence=0.0, status=''):
        """
        Build a unified Voice ID JSON payload and push it to the event queue.

        JSON schema:
        {
            "type":       str,   # wake | no_wake | speaker_verified | speaker_unverified |
                                 # stop_command | voice_id_active | timeout
            "speaker":    str,   # speaker id (e.g. speaker_0)
            "confidence": float, # cosine similarity score [0.0, 1.0]
            "status":     str    # VERIFIED | UNVERIFIED | ""
        }
        """
        payload = json.dumps({
            "type": event_type,
            "speaker": speaker,
            "confidence": round(confidence, 3),
            "status": status,
        })
        self.event_queue.put(('voice_id_event', payload))

    def _register_voice_id_callbacks(self):
        """
        Register callbacks for all Voice ID events via the EventManager.
        Patches VoiceIDAdapter.is_speaker_verified at class level to extract
        the cosine similarity confidence score from Speaker.voice_print embeddings.

        All events are published to /genai/voice_id/event as a unified JSON string.
        """
        try:
            from eiq_genai_flow.adapters.event_manager import EventType, Event
            from eiq_genai_flow.adapters.voice_id import VoiceIDAdapter

            # --- Cache for last confidence/speaker set by is_speaker_verified patch ---
            self._last_voice_id_confidence = 0.0
            self._last_voice_id_speaker = 'unknown'

            # --- Patch VoiceIDAdapter.is_speaker_verified at CLASS level ---
            # This is the only place where the similarity score is computed.
            # Speaker objects expose: voice_print, history_voice_print, id
            original_is_speaker_verified = VoiceIDAdapter.is_speaker_verified
            node_ref = self

            def is_speaker_verified_patched(adapter_self, audio_window):
                is_enrolled = original_is_speaker_verified(adapter_self, audio_window)

                try:
                    current_speaker = adapter_self.speaker_encoder(audio_window)
                    best_confidence = 0.0
                    best_speaker = 'unknown'

                    for i, spk in enumerate(adapter_self.speakers):
                        spk_vec = np.array(spk.voice_print).flatten()
                        cur_vec = np.array(current_speaker.voice_print).flatten()
                        norm = np.linalg.norm(spk_vec) * np.linalg.norm(cur_vec)
                        if norm > 1e-8:
                            sim = float(np.dot(spk_vec, cur_vec) / norm)
                            sim = max(0.0, sim)  # clamp to [0, 1]
                            if sim > best_confidence:
                                best_confidence = sim
                                best_speaker = f'speaker_{i}'

                    node_ref._last_voice_id_confidence = best_confidence
                    node_ref._last_voice_id_speaker = best_speaker

                except Exception as e:
                    node_ref.get_logger().error(f'[VoiceID] confidence extraction error: {e}')

                return is_enrolled

            VoiceIDAdapter.is_speaker_verified = is_speaker_verified_patched
            self.get_logger().info('[VoiceID] is_speaker_verified patched at class level')

            # --- Subscribe to all real Voice ID EventTypes ---
            voice_id_event_map = {
                'VOICE_ID_WAKE': ('wake', 'VERIFIED'),
                'VOICE_ID_NO_WAKE': ('no_wake', 'UNVERIFIED'),
                'VERIFIED_SPEAKER': ('speaker_verified', 'VERIFIED'),
                'UNVERIFIED_SPEAKER': ('speaker_unverified', 'UNVERIFIED'),
                'VOICE_ID_STOP_COMMAND': ('stop_command', 'UNVERIFIED'),
                'VOICE_ID_USED': ('voice_id_active', ''),
                'TIMEOUT': ('timeout', ''),
            }

            for event_type_name, (event_label, event_status) in voice_id_event_map.items():
                if hasattr(EventType, event_type_name):
                    event_type = getattr(EventType, event_type_name)

                    def make_cb(label, status):
                        def cb(event: Event):
                            self._voice_id_event(
                                label,
                                speaker=self._last_voice_id_speaker,
                                confidence=self._last_voice_id_confidence,
                                status=status,
                            )
                        return cb

                    self.genai_flow.event_manager.subscribe(event_type, make_cb(event_label, event_status))
                    self.get_logger().info(f'[VoiceID] subscribed to: {event_type_name} -> {event_label}')
                else:
                    self.get_logger().warning(f'[VoiceID] EventType NOT found: {event_type_name}')

            self.get_logger().info('[VoiceID] callbacks registered successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to register Voice ID callbacks: {e}')
            self.get_logger().error(traceback.format_exc())

    def _run_genai_flow_loop(self):
        """Run the GenAI Flow main loop in a background thread."""
        try:
            self.get_logger().info('Starting eIQ GenAI Flow main loop...')
            self.genai_flow.run()
            self.get_logger().info('eIQ GenAI Flow main loop exited normally')
        except Exception as e:
            self.get_logger().error(f'Error in GenAI Flow run loop: {e}')
            self.get_logger().error(traceback.format_exc())

    def _trigger_listening_callback(self, request, response):
        """Service callback: simulate an Enter key press to trigger STT listening."""
        self.get_logger().info('Trigger listening service called')
        try:
            self.stdin_queue.put('\n')
            response.success = True
            response.message = 'Listening triggered successfully'
        except Exception as e:
            self.get_logger().error(f'Failed to trigger listening: {e}')
            response.success = False
            response.message = str(e)
        return response

    def _text_input_callback(self, msg):
        """Topic callback: receive a text query and forward it to the stdin queue."""
        text = msg.data.rstrip('\n')
        self.get_logger().info(f'Received text input: {repr(text)}')
        try:
            self.stdin_queue.put(text)
        except Exception as e:
            self.get_logger().error(f'Failed to process text input: {e}')

    def _process_event_queue(self):
        """Drain the event queue and publish each event to the appropriate ROS 2 topic."""
        try:
            while not self.event_queue.empty():
                event_type, data = self.event_queue.get_nowait()

                if event_type == 'wakeword':
                    msg = " | ".join([f"{k}={v}" for k, v in data.items()])
                    self.wakeword_pub.publish(String(data=msg))
                    self.get_logger().info(f'Wakeword detected: {data["name"]} (energy: {data["energy"]:.2f})')

                elif event_type == 'vad_state':
                    self.vad_pub.publish(String(data=f"speech_detected: {data}"))
                    self.get_logger().info(f'VAD: {"Speech detected" if data else "Speech ended"}')

                elif event_type == 'stt_transcription':
                    self.stt_trans_pub.publish(String(data=str(data)))
                    self.get_logger().info(f'STT transcription: {data}')

                elif event_type == 'rag_category':
                    self.rag_cat_pub.publish(String(data=str(data)))
                    self.get_logger().info(f'RAG category: {data}')

                elif event_type == 'rag_answer':
                    self.rag_answer_pub.publish(String(data=str(data)))
                    self.get_logger().info(f'RAG answer: {data}')

                elif event_type == 'llm_token':
                    self.llm_token_pub.publish(String(data=str(data)))

                elif event_type == 'llm_response':
                    self.llm_response_pub.publish(String(data=str(data)))
                    self.get_logger().info(f'LLM full response: {data}')

                elif event_type == 'tts_synthesis':
                    self.tts_event_pub.publish(String(data=str(data)))
                    self.get_logger().info(str(data))

                elif event_type == 'tts_playback':
                    msg = "TTS playback started" if not data else "TTS playback ended"
                    self.tts_event_pub.publish(String(data=msg))
                    self.get_logger().info(msg)

                elif event_type == 'voice_id_event':
                    self.voice_id_event_pub.publish(String(data=str(data)))
                    self.get_logger().info(f'Voice ID event: {data}')

        except queue.Empty:
            pass
        except Exception as e:
            self.get_logger().error(f'Error processing event queue: {e}')

    def shutdown(self):
        """Gracefully shut down the GenAI Flow pipeline and restore stdin."""
        self.get_logger().info('Shutting down eIQ GenAI Flow...')
        if self.genai_flow and hasattr(self.genai_flow, 'shutdown'):
            self.genai_flow.shutdown()
        sys.stdin = self.original_stdin
        self.get_logger().info('Shutdown complete')


def main(args=None):
    try:
        rclpy.init(args=args)
        flow_node = GenAIFlowNode()
        executor = MultiThreadedExecutor()
        executor.add_node(flow_node)

        try:
            executor.spin()
        except KeyboardInterrupt:
            flow_node.get_logger().info('Keyboard interrupt received')
        finally:
            flow_node.shutdown()
            executor.shutdown()
            flow_node.destroy_node()

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
