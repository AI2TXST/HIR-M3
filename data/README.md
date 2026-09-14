# Data Access & Replication Guide

## CMS OASIS & Social Determinants of Health (SDoH) Data

Due to the Centers for Medicare & Medicaid Services (CMS) Data Use Agreement (DUA) and HIPAA regulations for Protected Health Information (PHI), individual-level patient records cannot be publicly distributed in this repository.

### Accessing the Datasets

1. **CMS OASIS Assessments**:
   - Researchers can request access to the OASIS national dataset through the **Research Data Assistance Center (ResDAC)** at [https://resdac.org](https://resdac.org).
   - Approval requires an active CMS DUA and institutional review board (IRB) review.

2. **Social Determinants of Health (SDoH)**:
   - Tract- and ZIP-level SDoH indicators are derived from the **AHRQ Social Determinants of Health Database** ([https://www.ahrq.gov/sdoh](https://www.ahrq.gov/sdoh)) and the **University of Wisconsin Area Deprivation Index (ADI)** ([https://www.neighborhoodatlas.medicine.wisc.edu](https://www.neighborhoodatlas.medicine.wisc.edu)).

3. **Geographic Boundaries & Rurality**:
   - Rural-Urban Commuting Area (RUCA) codes and standard ZIP-to-tract crosswalks are available via the **USDA Economic Research Service (ERS)** and the **U.S. Census Bureau**.

### Expected Directory Layout

Once downloaded and linked, place the processed files under `data/`:

```
data/
├── processed_final_mergedDF_condensed.csv      # Nationwide cohort (~100k rows)
├── processed_final_mergedDF_condensed_TX.csv   # Texas statewide cohort (~50k rows)
├── oasis_clusteredICD.csv                      # ICD-10 diagnostic hierarchy mappings
├── sdoh_2020_tract_1_0.csv                     # Census tract-level SDoH features
└── uszips.csv                                  # Geographic ZIP crosswalk
```
