from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from xurcroh.errors import CompatibilityError


def mux_video_audio(
    video: bytes,
    pcm_audio: bytes,
    *,
    sample_rate: int = 16_000,
    channels: int = 2,
) -> bytes:
    if sample_rate < 1 or channels < 1 or not video or not pcm_audio:
        raise ValueError("video, audio, sample rate, and channels are required")
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        raise CompatibilityError("audio/video muxing requires imageio-ffmpeg") from error
    with tempfile.TemporaryDirectory(prefix="xurcroh-av-") as directory:
        root = Path(directory)
        video_path = root / "input.mp4"
        audio_path = root / "output.mp4"
        video_path.write_bytes(video)
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(audio_path),
        ]
        result = subprocess.run(  # noqa: S603 - fixed ffmpeg executable, no shell
            command, input=pcm_audio, capture_output=True, check=False
        )
        if result.returncode != 0 or not audio_path.is_file():
            raise CompatibilityError(f"ffmpeg audio/video muxing failed: {result.stderr.decode()}")
        return audio_path.read_bytes()
