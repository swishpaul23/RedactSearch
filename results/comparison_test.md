# Detector comparison: `test` split

| detector | type | precision | recall | F1 | leak rate | support |
|---|---|---:|---:|---:|---:|---:|
| regex | PERSON | 0.000 | 0.000 | 0.000 | 100.0% | 266 |
| regex | EMAIL | 1.000 | 1.000 | 1.000 | 0.0% | 66 |
| regex | PHONE | 1.000 | 1.000 | 1.000 | 0.0% | 69 |
| regex | CARD | 1.000 | 1.000 | 1.000 | 0.0% | 43 |
| regex | SIN | 1.000 | 1.000 | 1.000 | 0.0% | 16 |
| regex | ACCOUNT | 1.000 | 1.000 | 1.000 | 0.0% | 24 |
| regex | overall | 1.000 | 0.450 | 0.621 | 55.0% | 484 |
| ner | PERSON | 0.924 | 0.865 | 0.893 | 14.3% | 266 |
| ner | EMAIL | 0.000 | 0.000 | 0.000 | 100.0% | 66 |
| ner | PHONE | 0.000 | 0.000 | 0.000 | 100.0% | 69 |
| ner | CARD | 0.000 | 0.000 | 0.000 | 100.0% | 43 |
| ner | SIN | 0.000 | 0.000 | 0.000 | 100.0% | 16 |
| ner | ACCOUNT | 0.000 | 0.000 | 0.000 | 100.0% | 24 |
| ner | overall | 0.924 | 0.475 | 0.628 | 52.9% | 484 |
| pipeline | PERSON | 0.924 | 0.865 | 0.893 | 14.3% | 266 |
| pipeline | EMAIL | 1.000 | 1.000 | 1.000 | 0.0% | 66 |
| pipeline | PHONE | 1.000 | 1.000 | 1.000 | 0.0% | 69 |
| pipeline | CARD | 1.000 | 1.000 | 1.000 | 0.0% | 43 |
| pipeline | SIN | 1.000 | 1.000 | 1.000 | 0.0% | 16 |
| pipeline | ACCOUNT | 1.000 | 1.000 | 1.000 | 0.0% | 24 |
| pipeline | overall | 0.959 | 0.926 | 0.942 | 7.9% | 484 |
| presidio | PERSON | 0.807 | 0.910 | 0.855 | 9.4% | 266 |
| presidio | EMAIL | 1.000 | 1.000 | 1.000 | 0.0% | 66 |
| presidio | PHONE | 0.896 | 1.000 | 0.945 | 0.0% | 69 |
| presidio | CARD | 1.000 | 1.000 | 1.000 | 0.0% | 43 |
| presidio | SIN | 0.000 | 0.000 | 0.000 | 56.2% | 16 |
| presidio | ACCOUNT | 0.362 | 0.708 | 0.479 | 0.0% | 24 |
| presidio | overall | 0.820 | 0.903 | 0.859 | 7.0% | 484 |
| presidio_casin | PERSON | 0.807 | 0.910 | 0.855 | 9.4% | 266 |
| presidio_casin | EMAIL | 1.000 | 1.000 | 1.000 | 0.0% | 66 |
| presidio_casin | PHONE | 0.885 | 1.000 | 0.939 | 0.0% | 69 |
| presidio_casin | CARD | 1.000 | 1.000 | 1.000 | 0.0% | 43 |
| presidio_casin | SIN | 1.000 | 0.750 | 0.857 | 0.0% | 16 |
| presidio_casin | ACCOUNT | 0.372 | 0.667 | 0.478 | 0.0% | 24 |
| presidio_casin | overall | 0.827 | 0.926 | 0.873 | 5.2% | 484 |

## Pipeline vs presidio

| type | pipeline recall | presidio recall | pipeline leak | presidio leak |
|---|---:|---:|---:|---:|
| CARD | 1.000 | 1.000 | 0.0% | 0.0% |
| SIN | 1.000 | 0.000 | 0.0% | 56.2% |
| PERSON | 0.865 | 0.910 | 14.3% | 9.4% |
| overall | 0.926 | 0.903 | 7.9% | 7.0% |

## Pipeline vs presidio_casin

| type | pipeline recall | presidio_casin recall | pipeline leak | presidio_casin leak |
|---|---:|---:|---:|---:|
| CARD | 1.000 | 1.000 | 0.0% | 0.0% |
| SIN | 1.000 | 0.750 | 0.0% | 0.0% |
| PERSON | 0.865 | 0.910 | 14.3% | 9.4% |
| overall | 0.926 | 0.926 | 7.9% | 5.2% |
