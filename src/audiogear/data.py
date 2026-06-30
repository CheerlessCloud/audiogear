from dataclasses import dataclass, field

from pydub.utils import mediainfo


@dataclass
class AudioSegment:
    id: str
    audio_file: str
    format: str
    path: str | None = None # this is a place holder to store the relative path of the audio file
    sample_rate: int | None = None
    channels: int | None = None
    bit_rate: str | None = None
    duration: float | None = None
    text: str | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __len__(self) -> float:
        return self.duration

    def print_labels(self):
        print(self.metadata.keys())

    def get_metric(self, metric: str) -> str | int | float | bool:
        return self.metadata[metric]

    def get_audio_info(self):
        info = mediainfo(self.audio_file)
        try:
            self.duration = float(info["duration"])
            self.sample_rate = int(info["sample_rate"])
            self.bit_rate = info["bit_rate"]
            self.format = info["format_name"]
            self.channels = int(info["channels"])
        except Exception as e:
            print(f"Error getting audio info for {self.audio_file}: {e}")


AudioPipeline = list[
    AudioSegment
]
