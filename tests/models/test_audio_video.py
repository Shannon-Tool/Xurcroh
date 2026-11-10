from __future__ import annotations

import torch

from xurcroh.models.av import mux_video_audio


def test_audio_video_muxing_produces_mp4() -> None:
    import io

    import imageio.v2 as imageio

    video = io.BytesIO()
    frame = torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()
    with imageio.get_writer(video, format="mp4", fps=2, codec="libx264") as writer:
        writer.append_data(frame)
        writer.append_data(frame)
    pcm = b"\x00\x00" * 16000
    output = mux_video_audio(video.getvalue(), pcm, sample_rate=16000, channels=1)
    assert b"ftyp" in output[:32]
    assert len(output) > len(pcm) / 100
