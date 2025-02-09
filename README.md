# ATSPM Aggregation

`atspm` is a lightweight production-ready Python package to efficiently transform traffic signal controller event logs into Aggregate Traffic Signal Performance Measures. 

This project focuses only on transforming event logs into performance measures and troubleshooting data, it does include data visualization. Feel free to submit feature requests or bug reports or to reach out with questions or comments. Contributions are welcome! 

## Features

- Transforms event logs into aggregate performance measures and troubleshooting metrics
- Supports incremental processing for real-time data (ie. every 15 minutes)
- Runs locally using the powerful [DuckDB](https://duckdb.org/) analytical SQL engine.
- Output to user-defined folder structure and file format (csv/parquet/json), or query DuckDB tables directly
- Deployed in production by Oregon DOT since July 2024

## Installation

```bash
pip install atspm
```
Or pinned to a specific version:
```bash
pip install atspm==1.x.x 
```
`atspm` works on Python 3.10-3.12 and is tested on Ubuntu, Windows, and MacOS.

## Quick Start

The best place to start is with these self-contained example uses in Colab!<br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14SPXPjpwbBEPpjKBN5s4LoqtHWSllvip?usp=sharing)

## Example Usage

Here's an example of how to use `atspm` (but see the Colab link above for more examples):

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

### Version 1.8.4 (September 12, 2024)

#### Bug Fixes / Improvements:
Fixed a timestamp conversion issue when reading unmatched events from a csv file. Updated the unit tests to catch this issue in the future. 

### Version 1.8.3 (September 5, 2024)

#### Bug Fixes / Improvements:
- Fixed estimated volumes for full_ped. Previously, it was converting 15-minute ped data to hourly by applying a rolling sum, then applying the quadratic transform to get volumes, and then converted back to 15-minute by undoing the rolling sum. The bug had to do with the data not always being ordered correctly before undoing the rolling sum. However, this update removes the undo rolling sum altogether and replaces it with multiplying hourly volumes by the ratio of 15-minute data to hourly data (more detail coming in the docs eventually). It seems to work much better now.

### Version 1.8.2 (August 29, 2024)

#### Bug Fixes / Improvements:
- Fixed issue when passing unmatched events as a dataframe instead of a file path.
- Added more tests for incremental runs when using dataframes. This is to mimic the ODOT production environment.

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

Ideas and contributions are welcome! Please feel free to submit a Pull Request. Note that GitHub Actions will automatically run unit tests on your code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
