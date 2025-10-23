# Research Sample: Mining ESG & Sustainable Capital
## Complete Pipeline Structure

### Part 1: Data Construction (01_data_construction/)
```
├── build_mining_panel.py          # Main data pipeline
├── match_snl_esg.py                # Adapt from firm_esg_join_by_name.py
├── extract_feasibility_dates.py   # Adapt from feasibility_extractor.py
└── geocode_mines.py                # NEW: Extract lat/lon from mine locations
```

**Output**: `mining_panel.csv`