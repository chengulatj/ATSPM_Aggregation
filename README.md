# ATSPM Aggregation

`atspm` is a production-ready Python package to transform hi-res ATC signal controller data into aggregate ATSPMs (Automated Traffic Signal Performance Measures). It runs locally using the powerful & lightweight [DuckDB](https://duckdb.org/) SQL engine.

## Features

- Transforms hi-res ATC signal controller data into aggregate ATSPMs
- Supports incremental processing for real-time data (ie. every 15 minutes)
- Output to user-defined folder structure and file format (csv/parquet/json), or query DuckDB tables directly
- Deployed in production by Oregon DOT since July 2024

## Installation

```bash
pip install atspm
```
Or pinned to a specific version:
```bash
pip install atspm==1.7.0 
```
`atspm` works on Python 3.10-3.12 and is tested on Ubuntu, Windows, and MacOS.

## Quick Start

Try out a self-contained example in Colab!<br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14SPXPjpwbBEPpjKBN5s4LoqtHWSllvip?usp=sharing)

## Detailed Usage

Here's a simple example of how to use `atspm`:

```python
# Import libraries
from atspm import SignalDataProcessor, sample_data

params = {
    # Global Settings
    'raw_data': sample_data.data, # dataframe or file path
    'detector_config': sample_data.config,
    'bin_size': 15, # in minutes
    'output_dir': 'test_folder',
    'output_to_separate_folders': True,
    'output_format': 'csv', # csv/parquet/json
    'output_file_prefix': 'prefix_',
    'remove_incomplete': True, # Remove periods with incomplete data
    'unmatched_event_settings': { # For incremental processing
        'df_or_path': 'test_folder/unmatched.parquet',
        'split_fail_df_or_path': 'test_folder/sf_unmatched.parquet',
        'max_days_old': 14},
    'to_sql': False, # Returns SQL string
    'verbose': 1, # 0: print off, 1: print performance, 2: print all
    # Performance Measures
    'aggregations': [
        {'name': 'has_data', 'params': {'no_data_min': 5, 'min_data_points': 3}},
        {'name': 'actuations', 'params': {}},
        {'name': 'arrival_on_green', 'params': {'latency_offset_seconds': 0}},
        {'name': 'communications', 'params': {'event_codes': '400,503,502'}},# MAXVIEW Specific
        {'name': 'coordination', 'params': {}},  # MAXTIME Specific
        {'name': 'ped', 'params': {}},
        {'name': 'unique_ped', 'params': {'seconds_between_actuations': 15}},
        {'name': 'full_ped', 'params': {
            'seconds_between_actuations': 15,
            'return_volumes': True
        }},
        {'name': 'split_failures', 'params': {
            'red_time': 5,
            'red_occupancy_threshold': 0.80,
            'green_occupancy_threshold': 0.80,
            'by_approach': True,
            'by_cycle': False
        }},
        {'name': 'splits', 'params': {}}, # MAXTIME Specific
        {'name': 'terminations', 'params': {}},
        {'name': 'yellow_red', 'params': {
            'latency_offset_seconds': 1.5,
            'min_red_offset': -8
        }},
        {'name': 'timeline', 'params': {'min_duration': 0.2, 'cushion_time': 60}},
    ]
}

processor = SignalDataProcessor(**params)
processor.run()
```

## Output Structure

After running the `SignalDataProcessor`, the output directory will have the following structure:

```
test_folder/
unmatched.parquet
sf_unmatched.parquet
├── actuations/
├── yellow_red/
├── arrival_on_green/
├── coordination/
├── terminations/
├── split_failures/
...etc...
```

Inside each folder, there will be a CSV file named `prefix_.csv` with the aggregated performance data. In production, the prefix could be named using the date/time of the run. Or you can output everything to a single folder.

A good way to use the data is to output as parquet to separate folders, and then a data visualization tool like Power BI can read in all the files in each folder and create a dashboard. For example, see: [Oregon DOT ATSPM Dashboard](https://app.powerbigov.us/view?r=eyJrIjoiNzhmNTUzNDItMzkzNi00YzZhLTkyYWQtYzM1OGExMDk3Zjk1IiwidCI6IjI4YjBkMDEzLTQ2YmMtNGE2NC04ZDg2LTFjOGEzMWNmNTkwZCJ9)

Use of CSV files in production should be avoided, instead use [Parquet](https://parquet.apache.org/) file format, which is significantly faster, smaller, and enforces datatypes.

## Performance Measures

The following performance measures are included:

- Actuations
- Arrival on Green
- Communications (MAXVIEW Specific, otherwise "Has Data" tells when controller generated data)
- Coordination (MAXTIME Specific)
- Detector Health
- Pedestrian Actuations, Services, and Estimated Volumes
- Split Failures
- Splits (MAXTIME Specific)
- Terminations
- Timeline Events
- Yellow and Red Actuations

*Coming Soon:*
- Total Pedestrian Delay
- Pedestrian Detector Health

Detailed documentation for each measure is coming soon.

## Release Notes

### Version 1.8.0 (August 28, 2024)

#### Bug Fixes / Improvements:
- Removed unused code from yellow_red for efficiency, but it's still not passing tests for incremental processing.

#### New Features:
- Added special functions and advance warning to timeline events.

### Version 1.7.0 (August 22, 2024)

#### Bug Fixes / Improvements:
- Fixed issue with incremental processing where cycles at the processing boundary were getting thrown out. This was NOT fixed yet for yellow_red!
- Significant changes to split_failures to make incremental processing more robust. For example, cycle timestamps are now tied to the end of the red period, not the start of the green period. 

#### New Features:
- Support for incremental processing added for split_failures & arrival_on_green. (yellow_red isn't passing tests yet)
- Added phase green, yellow & all red to timeline. 

## Future Plans

- Integration with [Ibis](https://ibis-project.org/) for compatibility with any SQL backend.
- Implement use of detector distance to stopbar for Arrival on Green calculations.
- Develop comprehensive documentation for each performance measure.

## Contributing

Ideas and contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
