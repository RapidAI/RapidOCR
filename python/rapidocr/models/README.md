# Spanish model assets

The reduced Spanish character dictionaries ship with the repository because they
are part of the configuration, not downloads:

| file | characters | used by |
| --- | --- | --- |
| `ppocr_keys_es.txt` | 116 | current checkpoint, type router |
| `ppocr_keys_es_v1.txt` | 107 | earlier checkpoints, regression tests |

The weights are too large to version and are published as release assets:

| file | size | notes |
| --- | --- | --- |
| `spanish_rec_v18.onnx` | 7.7 MB | default checkpoint for `LangRec.ES` |
| `spanish_rec_v19.onnx` | 7.7 MB | alternative, used by `type_router.py` |

Download them into this directory, or point `Rec.model_path` at wherever they
live. Each checkpoint must be used with the dictionary it was trained on:
mixing them corrupts CTC decoding (see `tests/test_spanish_dict.py`).

Both are fine-tuned from `PP-OCRv5_latin_mobile_rec` with head surgery to the
reduced vocabulary. Measured against the stock Latin checkpoint on 3,079
held-out crops (44,947 characters, 14 document types):
CER micro 0.1745 -> 0.1476, macro 0.1412 -> 0.1238.
