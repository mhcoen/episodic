"""
Voice input loop management for Episodic CLI.

This module encapsulates the voice listener task, voice/keyboard input racing,
and voice idle timer management. Extracted from cli_main.py to keep file sizes
manageable.
"""

import asyncio
from typing import Optional

import typer

from episodic.config import config

# Phrases that trigger voice-off when spoken
VOICE_OFF_PHRASES = [
    "exit voice", "voice off", "stop voice", "disable voice",
    "turn off voice", "voice mode off", "stop listening",
]

# Sentinel returned by get_input() when voice-off was spoken
VOICE_OFF_SIGNAL = "__voice_off__"


async def _voice_listener_task(
    voice_queue: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    """
    Async task that listens for voice input and puts results in a queue.

    Runs the blocking voice listen() in an executor to avoid blocking the
    event loop.
    """
    try:
        from episodic.voice import get_voice_manager

        manager = get_voice_manager()
        if not manager.is_active:
            manager.start()

        loop = asyncio.get_event_loop()

        while not stop_event.is_set():
            # Run blocking listen() in executor
            text = await loop.run_in_executor(
                None, lambda: manager.listen(timeout=2.0)
            )

            if text:
                # Check for sleep commands
                if manager.is_sleep_command(text):
                    manager.force_idle()
                    continue  # Don't put sleep command in queue

                await voice_queue.put(text)
                return  # Got input, exit task

            # Small yield to allow other tasks to run
            await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        typer.secho(f"Voice input error: {e}", fg="red")


class VoiceLoopManager:
    """Manages voice listener lifecycle and voice/keyboard input racing."""

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Listener lifecycle
    # ------------------------------------------------------------------

    def start_listener(self) -> None:
        """Start the voice listener task if not already running."""
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(
                _voice_listener_task(self._queue, self._stop_event)
            )

    def stop_listener(self) -> None:
        """Stop the voice listener task."""
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    # ------------------------------------------------------------------
    # Input acquisition
    # ------------------------------------------------------------------

    async def get_input(self, session) -> Optional[str]:
        """
        Get user input, racing voice and keyboard when voice mode is on.

        Returns:
            The user's text input, or
            VOICE_OFF_SIGNAL if the user spoke a voice-off phrase, or
            None if no input was captured (caller should ``continue``).
        """
        if config.get("voice_mode", False):
            return await self._get_voice_input(session)
        else:
            # Not in voice mode - stop any running listener
            self.stop_listener()
            return await session.prompt_async()

    async def _get_voice_input(self, session) -> Optional[str]:
        """Race between voice input and keyboard input."""
        self.start_listener()

        prompt_task = asyncio.create_task(session.prompt_async())

        wait_tasks = (
            [prompt_task, self._task] if self._task else [prompt_task]
        )
        done, pending = await asyncio.wait(
            wait_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        user_input: Optional[str] = None

        for task in done:
            if task is prompt_task:
                user_input = task.result()
                self.stop_listener()
            elif task is self._task:
                # Voice task completed - get from queue
                try:
                    user_input = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                # Restart voice listener for next input
                self._task = None

        if user_input is None:
            return None

        # Check for voice exit command (spoken)
        input_lower = user_input.lower().strip()
        if any(phrase in input_lower for phrase in VOICE_OFF_PHRASES):
            from episodic.commands.voice import voice_off

            self.stop_listener()
            voice_off()
            return VOICE_OFF_SIGNAL

        return user_input

    # ------------------------------------------------------------------
    # Idle timer helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pause_idle_timer() -> None:
        """Pause the voice idle timer during processing."""
        if config.get("voice_mode", False):
            from episodic.voice import get_voice_manager

            get_voice_manager().pause_idle_timer()

    @staticmethod
    def resume_idle_timer() -> None:
        """Resume the voice idle timer after processing."""
        if config.get("voice_mode", False):
            from episodic.voice import get_voice_manager

            get_voice_manager().resume_idle_timer()
