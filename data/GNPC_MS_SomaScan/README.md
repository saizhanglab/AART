# GNPC Paired Proteomics Data (MS ↔ SomaScan)

Paired mass spectrometry (MS) and SomaScan measurements from the GNPC cohort.

## Expected files

| File | Description |
|------|-------------|
| `ms_overlap.csv` | MS protein measurements (samples x proteins), `sample_id` as sample ID |
| `somascan_overlap.csv` | SomaScan aptamer measurements (samples x aptamers), `sample_id` as sample ID |
| `ms_feature_metadata.csv` | MS feature metadata with columns: `seqid_raw`, `probe_id`, `gene_name`, `uniprot`, `description` |

## Annotation

The annotation table is built automatically by `stage_data_aart.py` from the MS feature metadata.

```bash
python scripts/data_processing/stage_data_aart.py --config configs/aart/ms_to_soma.yaml
```
