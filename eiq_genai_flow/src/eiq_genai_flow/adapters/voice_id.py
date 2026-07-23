# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import logging
import threading
from collections import deque
from dataclasses import dataclass
from eiq_genai_flow.adapters.base import BaseAdapter
from voice_id.speaker import SpeakerEncoder, compare_speakers, merge_speakers
from voice_id.utils import consume_buffer
from audio_manager.audio_manager_base import ReaderConfig
from eiq_genai_flow.adapters.event_manager import EventManager, EventType, Event

logger = logging.getLogger(__name__)


@dataclass
class VoiceIDConfig:
    """Configuration for Voice id Adapter."""
    model_name = 'resnet34'
    # Various configuration
    audio_chunk_duration: int = 3
    max_registered_users: int = 1
    inactivity_timeout: float = 40.0
    # Maximum number of unverified chunk tolerated during a command,
    # once the speaker has already been identified as authorized.
    threshold_nb_chunk_unverified: int = 2


class VoiceIDAdapter(BaseAdapter):
    def __init__(self,
                 config,
                 audio_manager,
                 event_manager: EventManager,
                 vad_window_size_sec: int
                 ):
        super().__init__(audio_manager, event_manager)
        self.config = config

        # state management
        self._speech_start_event = threading.Event()
        self._speech_end_event = threading.Event()
        self._state_lock = threading.Lock()

        # Speech processing state
        self.speech_start_index = None
        self.speech_end_index = None

        self.speaker_encoder = SpeakerEncoder(name=config.model_name)

        # fixed configuration
        self.sample_rate = self.speaker_encoder.model_config.sample_rate
        self.model_samples_required = self.speaker_encoder.model_config.model_required_samples
        self.audio_chunk_length = int(self.config.audio_chunk_duration * self.sample_rate)
        self.vad_window_size_sec = vad_window_size_sec
        self.window_size = int(self.vad_window_size_sec * self.sample_rate)

        # Register with AudioManager
        voice_id_config = ReaderConfig(
            channels=1,
            format="F32LE",
            channel_indices=[0],
        )

        self.audio_reader_enroll = self.audio_manager.register_reader(name="VOICE_ID_ENROLL", config=voice_id_config)
        self.audio_reader_verify = self.audio_manager.register_reader(name="VOICE_ID_VERIFY", config=voice_id_config)

        self.enrolled_speaker = None
        self.speakers = deque(maxlen=config.max_registered_users)

        logger.info("Voice ID adapter initialized successfully")

    def enable(self, sync_to_current=True):
        # clear
        self._speech_start_event.clear()
        self._speech_end_event.clear()
        self.speech_start_index = None
        self.speech_end_index = None
        self._inactivity_elapsed = 0.0

        self.audio_reader_enroll.enable(sync_to_current=sync_to_current)
        self.audio_reader_verify.enable(sync_to_current=sync_to_current)

        # Subscribe to VIT WAKE - _on_vit_wake will subscribe to STT_END
        self.subscribe(EventType.VIT_WAKE, self._on_wake)

        if self.speakers:
            # At least one speaker is already enrolled, directly subscribe to vad speech start to avoid wake word
            self.subscribe(EventType.VAD_SPEECH_START, self._on_vad_speech_start)

        # start worker loop only if there is at least one registered speaker
        super().enable(sync_to_current=sync_to_current, start_worker_loop=bool(self.speakers))

    def disable(self, timeout: float = 5.0):
        super().disable(timeout=timeout)
        self.audio_reader_enroll.disable()
        self.audio_reader_verify.disable()

        # clear
        self._speech_end_event.clear()
        self.speech_start_index = None
        self.speech_end_index = None

    def _on_wake(self, event: Event):
        # In case of vit wake, we registered a new speaker and wait end of stt to verify speaker
        logger.debug(f"_on_wake event from {event.source}")

        # Clear status of vad to avoid voice id continue
        self._speech_end_event.clear()
        self._speech_start_event.clear()

        self.enrolling_speaker(event)

        self.unsubscribe(EventType.VIT_WAKE, self._on_wake)
        self.subscribe(EventType.STT_END, self.on_stt_end)

    def enrolling_speaker(self, event: Event):
        # Registered the speaker who says the wake word
        # get start, end indexes from vit
        speech_start_index = event.data.get("speech_start")
        speech_end_index = event.data.get("speech_end")
        self.audio_reader_enroll.read_index = speech_start_index

        speech_to_process = self.audio_reader_enroll.read(num_samples=speech_end_index - speech_start_index)
        self.enrolled_speaker = self.speaker_encoder(speech_to_process)  # run model

        # Update deque speakers registered
        tempo_speakers = list(self.speakers)
        original_sp_count = len(tempo_speakers)
        tempo_speakers.append(self.enrolled_speaker)
        # To not loose information about enrolled speaker if the new one added is already registered
        if len(tempo_speakers) >= 2:
            # need at least 2 speakers to compare
            compare_speakers(tempo_speakers)
        # if tempo_speakers is longer than maxlen, the oldest speaker is removed automatically
        self.speakers = deque(tempo_speakers, maxlen=self.config.max_registered_users)
        speaker_id = self.get_speaker_id(self.enrolled_speaker)
        if len(tempo_speakers) > original_sp_count:
            print(f"Speaker {speaker_id} registered successfully!")
        else:
            print(f"Speaker {speaker_id} recognized.")

    def _on_vad_speech_start(self, event: Event):
        logger.debug(f"_on_vad_speech_start event from {event.source}")
        with self._state_lock:
            self.speech_start_index = event.data.get("speech_start")
            self.audio_reader_verify.read_index = self.speech_start_index
            self._speech_start_event.set()
            # enable VAD end, avoid another VAD_SPEECH_START while voice_id processes
            self.subscribe(EventType.VAD_SPEECH_END, self._on_vad_speech_end)
            self.unsubscribe(EventType.VAD_SPEECH_START, self._on_vad_speech_start)

    def _on_vad_speech_end(self, event: Event):
        logger.debug(f"_on_vad_speech_end event from {event.source}")

        with self._state_lock:
            self.speech_end_index = event.data.get("speech_end")
            logger.debug("_speech_end_event: set")
            self._speech_end_event.set()

            self.unsubscribe(EventType.VAD_SPEECH_END, self._on_vad_speech_end)

    # At the end of stt command, after wake up from VIT only (otherwise, it's handle by send_verification_speaker),
    # always send event VERIFIED_SPEAKER
    def on_stt_end(self, event: Event):
        logger.debug(f"on_stt_end event from {event.source}")
        # Send speaker verification event at end of speech after VIT wake only
        # We consider after wake word the speaker is always verified
        self.publish(EventType.VERIFIED_SPEAKER)

    def _worker_loop(self):
        # Initialize variables
        speech_to_process = deque()
        is_already_verified = False
        count_unverified = 0

        # Enable VAD only if a speaker is already registered, if not, voice id don't need vad and wait for vit wake
        if self.speakers:
            self.publish(EventType.VOICE_ID_USED)

        while not self._stop_event.is_set():

            # Handle Inactivity
            if not self._speech_start_event.wait(timeout=self.vad_window_size_sec):
                # Only count inactivity after a wake event has been received
                self._inactivity_elapsed += self.vad_window_size_sec

                if self._inactivity_elapsed >= self.config.inactivity_timeout:
                    logger.warning(f"Inactivity timeout ({self.config.inactivity_timeout}s). Speakers unregistered, "
                                   f"wake word required to register again.")
                    self.publish(EventType.TIMEOUT)
                    self._stop_event.set()
                    self.speakers.clear()  # delete history after timeout
                    return
                continue

            # Handle speech start, read samples
            samples = self.audio_reader_verify.read(self.window_size, blocking=True)
            speech_to_process.extend(samples)

            # Handle speech end
            if self._speech_end_event.wait(timeout=self.vad_window_size_sec):

                # get remaining data until speech_end_index
                window_size = self.speech_end_index - self.audio_reader_verify.read_index
                samples = self.audio_reader_verify.read(window_size, blocking=True)
                speech_to_process.extend(samples)

                is_enrolled = self.handle_speech_end(speech_to_process, is_already_verified, count_unverified)
                self.send_verification_speaker(is_enrolled)

                # Clean status
                is_already_verified = False
                count_unverified = 0
                speech_to_process.clear()
                self._speech_end_event.clear()
                self._speech_start_event.clear()

            # Handle speech if enough samples to process
            elif len(speech_to_process) >= self.audio_chunk_length:
                is_already_verified, count_unverified, is_enrolled = self.handle_during_speech(speech_to_process,
                                                                                               is_already_verified,
                                                                                               count_unverified)

    def handle_during_speech(self, speech_to_process, is_already_verified, count_unverified):

        chunk = consume_buffer(speech_to_process, self.audio_chunk_length)
        logger.info("handling during speech")
        is_enrolled = self.is_speaker_verified(chunk)

        # Speaker is enrolled and no wake up command has been sent to STT
        if is_enrolled and not is_already_verified:
            logger.debug(f"Sending audio of length {len(chunk)} to STT ")
            is_already_verified = True  # update to only wake up STT once
            # Send VOICE ID WAKE only once at the beginning of the speech
            self.publish(EventType.VOICE_ID_WAKE, data={"speech_start": self.speech_start_index,
                                                        "window_size": self.audio_chunk_length,
                                                        "is_speech_ended": False})

        # Speaker is not enrolled
        elif not is_enrolled:
            # During listening already authorized by voice id
            if is_already_verified:
                count_unverified += 1
                # When speaker is not recognized threshold_nb_chunk_unverified times during a command,
                # Interrupt processing of speech !
                if count_unverified >= self.config.threshold_nb_chunk_unverified:
                    logger.warning("Speaker unrecognized during command, interrupt processing")
                    self.publish(EventType.VOICE_ID_STOP_COMMAND)
                    self._stop_event.set()
                else :
                    is_enrolled = True  # don't change status of verified speaker

            # If it's the beginning of the speech, we don't want to register speech of unverified speaker
            else:
                speech_to_process.clear()  # reset speech to process
                self.publish(EventType.VOICE_ID_NO_WAKE)
                self._speech_start_event.clear()  # wait another vad
                self._speech_end_event.clear()  # wait another vad
                self.unsubscribe(EventType.VAD_SPEECH_END, self._on_vad_speech_end)
                self.subscribe(EventType.VAD_SPEECH_START, self._on_vad_speech_start)

        # update status
        return is_already_verified, count_unverified, is_enrolled

    def handle_speech_end(self, speech_to_process, is_already_verified, count_unverified):

        is_enrolled = False

        if len(speech_to_process) >= self.model_samples_required:
            # Ensure that there is enough samples required by the model
            logger.debug("handle_speech_end: consume_buffer")
            chunk = consume_buffer(speech_to_process, len(speech_to_process))

            is_enrolled = self.is_speaker_verified(chunk)

            # Check if the speaker is already enrolled from before
            if is_enrolled:
                # Send VOICE_ID_WAKE to not wait wake word and directly process this speech
                logger.debug(f"Sending audio of length {len(chunk)} to STT ")
                self.publish(EventType.VOICE_ID_WAKE, data={"speech_start": self.speech_start_index,
                                                            "speech_end": self.speech_end_index,
                                                            "is_speech_ended": True})

            else:
                count_unverified += 1
                if is_already_verified and count_unverified < self.config.threshold_nb_chunk_unverified:
                    # Allow speech if speech has already been allowed (in handle_during_speech)
                    # And nb of unverified command don't exceed max times

                    # Send VOICE_ID_WAKE to not wait wake word and directly process this speech
                    logger.debug(f"Sending audio of length {len(chunk)} to STT ")
                    self.publish(EventType.VOICE_ID_WAKE, data={"speech_start": self.speech_start_index,
                                                                "speech_end": self.speech_end_index,
                                                                "is_speech_ended": True})
                    is_enrolled = True

                elif is_already_verified and count_unverified >= self.config.threshold_nb_chunk_unverified:
                    # When speaker is not recognized threshold_nb_chunk_unverified times during a command,
                    # Interrupt processing of speech !
                    logger.warning("Speaker unrecognized during command, interrupt processing")
                    self.publish(EventType.VOICE_ID_STOP_COMMAND)
                    self._stop_event.set()

                else:
                    logger.info("Speaker unrecognized")
                    self.publish(EventType.VOICE_ID_NO_WAKE)
                    # Wait another vad start
                    self.subscribe(EventType.VAD_SPEECH_START, self._on_vad_speech_start)
        else :
            logger.info("Not enough samples to process speaker verification")
            self.publish(EventType.VOICE_ID_NO_WAKE)
            # Wait another vad start
            self.subscribe(EventType.VAD_SPEECH_START, self._on_vad_speech_start)

        return is_enrolled

    def send_verification_speaker(self, is_enrolled):
        if is_enrolled :
            self.publish(EventType.VERIFIED_SPEAKER)
        else:
            self.publish(EventType.UNVERIFIED_SPEAKER)

    def is_speaker_verified(self, audio_window):
        current_speaker = self.speaker_encoder(audio_window)
        is_enrolled = False
        for spk in self.speakers:
            if spk == current_speaker:
                # Keep history of the current speaker
                merge_speakers(spk, current_speaker)
                is_enrolled = True
                print(f"Speaker {self.get_speaker_id(spk)} recognized.")
        return is_enrolled

    def get_speaker_id(self, speaker):
        """Get speaker ID based on position in deque."""
        for idx, spk in enumerate(self.speakers):
            if spk.id is speaker.id:
                return idx + 1
