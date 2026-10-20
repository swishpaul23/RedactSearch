# Detector comparison: `dev` split

| detector | type | precision | recall | F1 | leak rate | support |
|---|---|---:|---:|---:|---:|---:|
| regex | PERSON | 0.000 | 0.000 | 0.000 | 100.0% | 154 |
| regex | EMAIL | 1.000 | 1.000 | 1.000 | 0.0% | 32 |
| regex | PHONE | 1.000 | 1.000 | 1.000 | 0.0% | 37 |
| regex | CARD | 1.000 | 1.000 | 1.000 | 0.0% | 31 |
| regex | SIN | 1.000 | 1.000 | 1.000 | 0.0% | 5 |
| regex | ACCOUNT | 1.000 | 1.000 | 1.000 | 0.0% | 10 |
| regex | overall | 1.000 | 0.428 | 0.599 | 57.2% | 269 |
| ner | PERSON | 0.933 | 0.909 | 0.921 | 9.1% | 154 |
| ner | EMAIL | 0.000 | 0.000 | 0.000 | 100.0% | 32 |
| ner | PHONE | 0.000 | 0.000 | 0.000 | 100.0% | 37 |
| ner | CARD | 0.000 | 0.000 | 0.000 | 100.0% | 31 |
| ner | SIN | 0.000 | 0.000 | 0.000 | 100.0% | 5 |
| ner | ACCOUNT | 0.000 | 0.000 | 0.000 | 100.0% | 10 |
| ner | overall | 0.933 | 0.520 | 0.668 | 48.0% | 269 |
| pipeline | PERSON | 0.933 | 0.909 | 0.921 | 9.1% | 154 |
| pipeline | EMAIL | 1.000 | 1.000 | 1.000 | 0.0% | 32 |
| pipeline | PHONE | 1.000 | 1.000 | 1.000 | 0.0% | 37 |
| pipeline | CARD | 1.000 | 1.000 | 1.000 | 0.0% | 31 |
| pipeline | SIN | 1.000 | 1.000 | 1.000 | 0.0% | 5 |
| pipeline | ACCOUNT | 1.000 | 1.000 | 1.000 | 0.0% | 10 |
| pipeline | overall | 0.962 | 0.948 | 0.955 | 5.2% | 269 |
| presidio | PERSON | 0.892 | 0.909 | 0.900 | 9.1% | 154 |
| presidio | EMAIL | 1.000 | 1.000 | 1.000 | 0.0% | 32 |
| presidio | PHONE | 0.974 | 1.000 | 0.987 | 0.0% | 37 |
| presidio | CARD | 1.000 | 1.000 | 1.000 | 0.0% | 31 |
| presidio | SIN | 0.000 | 0.000 | 0.000 | 60.0% | 5 |
| presidio | ACCOUNT | 0.474 | 0.900 | 0.621 | 0.0% | 10 |
| presidio | overall | 0.899 | 0.926 | 0.912 | 6.3% | 269 |
| presidio_casin | PERSON | 0.892 | 0.909 | 0.900 | 9.1% | 154 |
| presidio_casin | EMAIL | 1.000 | 1.000 | 1.000 | 0.0% | 32 |
| presidio_casin | PHONE | 0.974 | 1.000 | 0.987 | 0.0% | 37 |
| presidio_casin | CARD | 1.000 | 1.000 | 1.000 | 0.0% | 31 |
| presidio_casin | SIN | 1.000 | 1.000 | 1.000 | 0.0% | 5 |
| presidio_casin | ACCOUNT | 0.529 | 0.900 | 0.667 | 0.0% | 10 |
| presidio_casin | overall | 0.907 | 0.944 | 0.925 | 5.2% | 269 |

## Pipeline vs presidio

| type | pipeline recall | presidio recall | pipeline leak | presidio leak |
|---|---:|---:|---:|---:|
| CARD | 1.000 | 1.000 | 0.0% | 0.0% |
| SIN | 1.000 | 0.000 | 0.0% | 60.0% |
| PERSON | 0.909 | 0.909 | 9.1% | 9.1% |
| overall | 0.948 | 0.926 | 5.2% | 6.3% |

## Pipeline vs presidio_casin

| type | pipeline recall | presidio_casin recall | pipeline leak | presidio_casin leak |
|---|---:|---:|---:|---:|
| CARD | 1.000 | 1.000 | 0.0% | 0.0% |
| SIN | 1.000 | 1.000 | 0.0% | 0.0% |
| PERSON | 0.909 | 0.909 | 9.1% | 9.1% |
| overall | 0.948 | 0.944 | 5.2% | 5.2% |
