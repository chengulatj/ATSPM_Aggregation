# ATSPM Aggregation

`atspm` is a production-ready Python package to transform hi-res ATC signal controller data into aggregate ATSPMs (Automated Traffic Signal Performance Measures). It runs locally using the lightweight yet powerful [DuckDB](https://duckdb.org/) SQL engine.

## Features

- Transforms high-resolution ATC signal controller data into aggregate ATSPMs
- Uses DuckDB for efficient local processing
- Supports various output formats (CSV, Parquet, JSON)
- Includes multiple performance measures (Actuations, Arrival on Green, Split Failures, etc.)
- Deployed in production by Oregon DOT since July 2024

## Installation

```bash
pip install atspm
```
Or pinned to a specific version:
```bash
pip install atspm==1.6.0
```
`atspm` was developed with Python 3.11 and should work with 3.7 and up.

## Quick Start

Try out a self-contained example in Colab!<br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14SPXPjpwbBEPpjKBN5s4LoqtHWSllvip?usp=sharing)

## Detailed Usage

Here's a simple example of how to use `atspm`:

```python
# Import libraries
from atspm import SignalDataProcessor, sample_data

# Set up all parameters
params = {
    # Global Settings
    'raw_data': sample_data.data, # dataframe or file path to csv/parquet/json
    'detector_config': sample_data.config,
    'bin_size': 15, # in minutes
    'output_dir': 'test_folder',
    'output_to_separate_folders': True,
    'output_format': 'csv', # csv/parquet/json
    'output_file_prefix': 'test_prefix',
    'remove_incomplete': True, # Remove periods with incomplete data by joining to the has_data table
    # Performance Measures
    'aggregations': [
        {'name': 'has_data', 'params': {'no_data_min': 5, 'min_data_points': 3}}, # in minutes, ie remove bins with less than 3 rows every 5 minutes
        {'name': 'actuations', 'params': {}},
        {'name': 'arrival_on_green', 'params': {'latency_offset_seconds': 0}},
        {'name': 'communications', 'params': {'event_codes': '400,503,502'}}, # MAXVIEW Specific
        {'name': 'coordination', 'params': {}}, # MAXTIME Specific
        {'name': 'full_ped', 'params': {'seconds_between_actuations': 15, 'return_volumes':True}},
        {'name': 'split_failures', 'params': {'red_time': 5, 'red_occupancy_threshold': 0.80, 'green_occupancy_threshold': 0.70, 'by_approach': True}},
        {'name': 'splits', 'params': {}}, # MAXTIME Specific
        {'name': 'terminations', 'params': {}},
        {'name': 'yellow_red', 'params': {'latency_offset_seconds': 1.5, 'min_red_offset': -8}}, # min_red_offset is optional, it filters out actuations occuring -n seconds before start of red
        {'name': 'timeline', 'params': {'cushion_time':60, 'max_event_days': 14}},   
    ]
}

processor = SignalDataProcessor(**params)
processor.run()
```

## Output Structure

After running the `SignalDataProcessor`, the output directory will have the following structure:

```
test_folder/
├── actuations/
├── yellow_red/
├── arrival_on_green/
├── coordination/
├── terminations/
├── split_failures/
...etc...
```

Inside each folder, there will be a CSV file named `test_prefix.csv` with the aggregated performance data. The prefix can be used, for example, by setting it to the run date. Or you can output everything to a single folder.

A good way to use the data is to output as parquet to separate folders, and then a data visualization tool like Power BI can read in all the files in each folder and create a dashboard. For example, see: [Oregon DOT ATSPM Dashboard](https://app.powerbigov.us/view?r=eyJrIjoiNzhmNTUzNDItMzkzNi00YzZhLTkyYWQtYzM1OGExMDk3Zjk1IiwidCI6IjI4YjBkMDEzLTQ2YmMtNGE2NC04ZDg2LTFjOGEzMWNmNTkwZCJ9)

## Performance Measures

The following performance measures are included:

- Actuations
- Arrival on Green
- Communications (MaxView Specific, otherwise "Has Data" tells when controller generated data)
- Coordination (MaxTime Specific)
- Detector Health
- Pedestrian Actuations, Services, and Estimated Volumes
- Split Failures
- Splits (MaxTime Specific)
- Terminations
- Timeline Events
- Yellow and Red Actuations

*Coming Soon:*
- Total Pedestrian Delay
- Pedestrian Detector Health

Detailed documentation for each measure is coming soon.

## Release Notes

### Version 1.6.0 (July 29, 2024)

#### Bug Fixes / Improvements:
- Improved performance for arrival_on_green, split_failures, and yellow_red by filtering out unused phase events.
- Now using a stable version of DuckDB (1.0.0).
- Now saves unmatched events from timeline events, to be loaded in next time. This assumes running on a schedule like every 15 minutes.
- output_file_prefix is no longer required when saving.

#### New Features:
- Set a max number of days for unmatched events before removal to prevent accumulation over time.
- Added verbose option to set level of print/debug statements.
- `.to_sql()` method now returns SQL string instead of executing it, for debugging or executing on other platforms.

## Future Plans

- Integration with [Ibis](https://ibis-project.org/) for compatibility with any SQL backend.
- Implement use of detector distance to stopbar for Arrival on Green calculations.
- Develop comprehensive documentation for each performance measure.

## Contributing

Ideas and contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
