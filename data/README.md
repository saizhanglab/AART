# Data Directory

This directory is excluded from version control. Place your data files here following the expected structure.

## Expected Data Format

### CKB Olink-SomaScan paired data (training)

- `olink_overlap_train.csv` / `olink_overlap_test.csv` — Olink protein measurements (samples x proteins), with a `csid` column as sample identifier
- `somascan_overlap_train.csv` / `somascan_overlap_test.csv` — SomaScan aptamer measurements (samples x aptamers), matched by `csid`
- `somascan_annotation_cache.csv` — SomaScan annotation table mapping aptamer IDs to gene symbols (columns: `SeqId`, `Target`, `UniProt`, `EntrezGeneSymbol`)

### MS-Olink paired data (optional)

Place under `data/MS_Olink/method_ready/`:
- `olink_overlap_train.csv` / `olink_overlap_test.csv`
- `ms_overlap_train.csv` / `ms_overlap_test.csv`
- `ms_annotation_cache.csv`

### GNPC cohort data (for application/transfer learning)

Due to data access restrictions, GNPC analyses must be run on the appropriate secure platform. Upload the relevant scripts to the platform to perform the analysis.

### UKB cohort data (for disease prediction)

Due to data access restrictions, UKB analyses must be run on the appropriate secure platform (e.g., DNANexus). Upload the relevant scripts to the platform to perform the analysis.
