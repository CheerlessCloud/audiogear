import csv
import dataclasses
from io import StringIO
from typing import Callable

from loguru import logger

from audiogear.data import AudioSegment
from audiogear.io import DataFolderLike
from audiogear.pipeline.writers.base_disk import DiskWriter


class CsvWriter(DiskWriter):
    name = "🔢 Csv"
    default_output_filename: str = "metadata.csv"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: str | None = None,
        adapter: Callable = None,
        sep: str = "|",
    ):
        super().__init__(output_folder, output_filename=output_filename, compression=compression, adapter=adapter)
        self.sep = sep
        # The column layout is locked from the FIRST row written and reused for
        # every subsequent row and every output file. CSV is a positional format:
        # if rows carried different field sets (e.g. one clip missing `bit_rate`),
        # `csv.DictWriter` would emit a row with fewer columns than the header and
        # shift every value after the gap — so a neighbouring column's value
        # (e.g. `pyt_si_sdr=-30`) is read back under another header (`whisper_wer`).
        # That was the root cause of the impossible MOS/WER values in BUG-1.
        self.fieldnames: list[str] | None = None
        # One header per physical output file. A single writer instance can span
        # several files (``$rank`` templating, ``max_file_size`` rotation), so we
        # track which files already got a header instead of a single bool.
        self._headers_written: set[str] = set()

    def _default_adapter(self, segment: AudioSegment) -> dict:
        """CSV-specific adapter: keep EVERY field, even falsy ones.

        The base adapter drops falsy values (``if val``), which is fine for a
        sparse format like JSONL but makes CSV rows ragged (a clip with empty
        ``text`` or ``bit_rate=None`` would be missing those columns). Keeping all
        fields guarantees a stable schema; missing values serialise as empty
        cells. ``metadata`` stays nested here and is flattened in ``_write``.
        """
        data = dataclasses.asdict(segment)
        data.pop("path", None)  # only needed to remember the original relative path
        if self.expand_metadata and "metadata" in data:
            data |= data.pop("metadata")
        return data

    def _write(self, segment: dict, file_handler: StringIO, filename: str):
        metadata = segment.pop("metadata", {}) or {}
        flat_data = {**segment, **metadata}
        flat_data.pop("metadata", None)

        if self.fieldnames is None:
            self.fieldnames = list(flat_data.keys())
        unexpected = [k for k in flat_data if k not in self.fieldnames]
        if unexpected:
            # A later row introduced columns the locked header does not have.
            # Dropping them keeps the file aligned (better a missing column than
            # a corrupt one); warn so the schema drift is visible.
            logger.warning(
                f"CsvWriter: row {flat_data.get('id')!r} has columns absent from the "
                f"locked header and will be dropped to keep the CSV aligned: {unexpected}"
            )

        csv_output = StringIO()
        csv_writer = csv.DictWriter(
            csv_output,
            fieldnames=self.fieldnames,
            delimiter=self.sep,
            restval="",  # field missing from this row -> empty cell (preserve alignment)
            extrasaction="ignore",  # unexpected field -> drop it (preserve alignment)
        )
        if filename not in self._headers_written:
            csv_writer.writeheader()
            self._headers_written.add(filename)
        csv_writer.writerow(flat_data)
        file_handler.write(csv_output.getvalue())
