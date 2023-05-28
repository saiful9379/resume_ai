# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 NAVER CLOVA Team. All rights reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from transformers.file_utils import (
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)

_import_structure = {
    "configuration_bros": ["BROS_PRETRAINED_CONFIG_ARCHIVE_MAP", "BrosConfig"],
    "tokenization_bros": ["BrosTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_bros_fast"] = ["BrosTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_bros"] = [
        "BROS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BrosForMaskedLM",
        "BrosForPreTraining",
        "BrosForSequenceClassification",
        "BrosForTokenClassification",
        "BrosModel",
        "BrosLMHeadModel",
        "BrosPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_bros import BROS_PRETRAINED_CONFIG_ARCHIVE_MAP, BrosConfig
    from .tokenization_bros import BrosTokenizer

    if is_tokenizers_available():
        from .tokenization_bros_fast import BrosTokenizerFast

    if is_torch_available():
        from .modeling_bros import (
            BROS_PRETRAINED_MODEL_ARCHIVE_LIST,
            BrosForMaskedLM,
            BrosForPreTraining,
            BrosForSequenceClassification,
            BrosForTokenClassification,
            BrosLMHeadModel,
            BrosModel,
            BrosPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure
    )
