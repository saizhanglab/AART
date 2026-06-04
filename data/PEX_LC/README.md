# PEX-LC Paired Proteomics Data (MS ↔ Olink)

Paired mass spectrometry (MS) and Olink measurements from the PEX-LC cohort (n = 88).

The data is downloaded from:
https://doi.org/10.6084/m9.figshare.27931350

## Expected files

| File | Description |
|------|-------------|
| `ms_overlap.csv` | MS protein measurements (samples x proteins), `csid` as sample ID, columns in `seq.X.Y` format |
| `olink_overlap.csv` | Olink protein measurements (samples x proteins), `csid` as sample ID, columns as gene symbols |
| `ms_feature_metadata.csv` | MS feature metadata with columns: `seqid_raw`, `probe_id`, `gene_name`, `uniprot`, `description` |

## Annotation

The annotation table is built automatically by `stage_data_aart.py` from the MS feature metadata.
Each MS probe maps to a gene symbol via the `gene_name` column — no external mapping table is needed.

```bash
python scripts/data_processing/stage_data_aart.py --config configs/aart/ms_to_olink.yaml
```
