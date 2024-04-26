import json
from typing import List

import torch
from langchain.document_loaders.unstructured import UnstructuredFileLoader
import os
from typing import Union, Any
from faster_whisper import WhisperModel

from qanything_kernel.utils.custom_log import debug_logger


class UnstructuredPaddleAudioLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load audio files, such as mp3, wav, and mp4."""

    def __init__(
            self,
            file_path: Union[str, List[str]],
            model_name: str = "large-v3",
            use_cpu: bool = True,
            mode: str = "single",
            whisper_model: WhisperModel = None,
            **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        self.model_name = model_name
        if not model_name:
            self.model_name = "large-v3"

        self.use_cpu = use_cpu
        if not whisper_model:
            # TODO: CUDA memory may run out.
            if torch.cuda.is_available():
                debug_logger.warning("audio parser Using CUDA, may run out of memory.")
                self.whisper_model = WhisperModel(self.model_name, device="cuda", compute_type="float16")
            else:
                debug_logger.warning("audio parser Using CPU, may take a long time.")
                self.whisper_model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        debug_logger.info(f"Using Whisper model: {self.model_name},file_path: {file_path}")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        def audio_to_txt(filepath, dir_path="tmp_files"):
            full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
            if not os.path.exists(full_dir_path):
                os.makedirs(full_dir_path)
            filename = os.path.split(filepath)[-1]
            segments, info = self.whisper_model.transcribe(filepath, vad_filter=True)
            result = []
            for segment in segments:
                s = "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
                debug_logger.info(f"s: {s}")
                result.append(s)
            txt_file_path = os.path.join(full_dir_path, "%s.txt" % filename)
            with open(txt_file_path, 'w', encoding='utf-8') as fout:
                fout.write("\n".join(result))
            return txt_file_path

        txt_file_path = audio_to_txt(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(filename=txt_file_path, **self.unstructured_kwargs)
