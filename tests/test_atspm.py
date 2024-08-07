import pytest
import pandas as pd
import os
import shutil
from src.atspm import SignalDataProcessor, sample_data
import duckdb

# Define the parameters for testing
TEST_PARAMS = {
  'raw_data': sample_data.data,
  'detector_config': sample_data.config,
  'bin_size': 15,
  'output_dir': 'test_output',
  'output_to_separate_folders': False,
  'output_format': 'parquet',
  'output_file_prefix': 'test_',
  'remove_incomplete': True,
  'unmatched_events': None,
  'to_sql': False,
  'verbose': 0,
  'aggregations': [
      {'name': 'has_data', 'params': {'no_data_min': 5, 'min_data_points': 3}},
      {'name': 'actuations', 'params': {}},
      {'name': 'arrival_on_green', 'params': {'latency_offset_seconds': 0}},
      {'name': 'communications', 'params': {'event_codes': '400,503,502'}},
      {'name': 'coordination', 'params': {}},
      {'name': 'ped', 'params': {}},
      {'name': 'unique_ped', 'params': {'seconds_between_actuations': 15}},
      {'name': 'full_ped', 'params': {'seconds_between_actuations': 15, 'return_volumes':True}},
      {'name': 'split_failures', 'params': {'red_time': 5, 'red_occupancy_threshold': 0.80, 'green_occupancy_threshold': 0.80, 'by_approach': True}},
      {'name': 'splits', 'params': {}},
      {'name': 'terminations', 'params': {}},
      {'name': 'yellow_red', 'params': {'latency_offset_seconds': 1.5, 'min_red_offset': -8}},
      {'name': 'timeline', 'params': {'cushion_time':60, 'max_event_days': 14}},
  ]
}

# Define aggregations that can be run incrementally
INCREMENTAL_AGGREGATIONS = [agg for agg in TEST_PARAMS['aggregations'] if agg['name'] not in ['unique_ped', 'full_ped']]

@pytest.fixture(scope="module")
def processor_output():
  """Fixture to run the SignalDataProcessor once for all tests"""
  processor = SignalDataProcessor(**TEST_PARAMS)
  processor.run()
  yield
  # Cleanup after all tests are done
  shutil.rmtree(TEST_PARAMS['output_dir'])

def compare_dataframes(df1, df2):
  """Compare two dataframes, ignoring row order"""
  df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
  df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)
  pd.testing.assert_frame_equal(df1_sorted, df2_sorted)

@pytest.mark.parametrize("aggregation", TEST_PARAMS['aggregations'], ids=lambda x: x['name'])
def test_aggregation(processor_output, aggregation):
  """Test each aggregation individually"""
  agg_name = aggregation['name']
  output_file = os.path.join(TEST_PARAMS['output_dir'], f"{TEST_PARAMS['output_file_prefix']}{agg_name}.parquet")
  precalc_file = f"tests/precalculated/{agg_name}.parquet"

  assert os.path.exists(output_file), f"Output file for {agg_name} not found"
  assert os.path.exists(precalc_file), f"Precalculated file for {agg_name} not found"

  output_df = pd.read_parquet(output_file)
  precalc_df = pd.read_parquet(precalc_file)

  compare_dataframes(output_df, precalc_df)

def test_all_files_generated():
  """Test that all expected files are generated"""
  expected_files = [f"{TEST_PARAMS['output_file_prefix']}{agg['name']}.parquet" for agg in TEST_PARAMS['aggregations']]
  for file in expected_files:
      assert os.path.exists(os.path.join(TEST_PARAMS['output_dir'], file)), f"File {file} not generated"

@pytest.fixture(scope="module")
def incremental_processor_output():
  """Fixture to run the SignalDataProcessor incrementally"""
  d = duckdb.connect()
  data = sample_data.data

  chunks = {
      '1_chunk': d.sql("select * from data where timestamp >= '2024-04-15 12:00:00' and timestamp < '2024-04-15 12:15:00'").df(),
      '2_chunk': d.sql("select * from data where timestamp >= '2024-04-15 12:10:00' and timestamp < '2024-04-15 12:30:00'").df(),
      '3_chunk': d.sql("select * from data where timestamp >= '2024-04-15 12:25:00' and timestamp < '2024-04-15 12:45:00'").df(),
      '4_chunk': d.sql("select * from data where timestamp >= '2024-04-15 12:40:00' and timestamp < '2024-04-15 13:00:00'").df(),
      '5_chunk': d.sql("select * from data where timestamp >= '2024-04-15 12:55:00' and timestamp < '2024-04-15 13:15:00'").df(),
      '6_chunk': d.sql("select * from data where timestamp >= '2024-04-15 13:10:00' and timestamp < '2024-04-15 13:30:00'").df(),
      '7_chunk': d.sql("select * from data where timestamp >= '2024-04-15 13:25:00' and timestamp < '2024-04-15 13:45:00'").df(),
      '8_chunk': d.sql("select * from data where timestamp >= '2024-04-15 13:40:00' and timestamp < '2024-04-15 14:00:00'").df()
  }

  output_dir = 'test_incremental_output'
  os.makedirs(output_dir, exist_ok=True)

  for i, chunk in chunks.items():
      params = TEST_PARAMS.copy()
      params.update({
          'raw_data': chunk,
          'output_dir': output_dir,
          'output_file_prefix': f"{i}_",
          'aggregations': INCREMENTAL_AGGREGATIONS  # Use only incremental aggregations
      })
      processor = SignalDataProcessor(**params)
      processor.run()

  yield output_dir
  # Cleanup after all tests are done
  shutil.rmtree(output_dir)

@pytest.mark.parametrize("aggregation", INCREMENTAL_AGGREGATIONS, ids=lambda x: x['name'])
def test_incremental_aggregation(incremental_processor_output, aggregation):
  """Test each aggregation for incremental runs"""
  agg_name = aggregation['name']
  output_files = [os.path.join(incremental_processor_output, f"{i}_chunk_{agg_name}.parquet") for i in range(1, 9)]
  precalc_file = f"tests/precalculated/{agg_name}.parquet"

  for file in output_files:
      assert os.path.exists(file), f"Incremental output file {file} not found"
  assert os.path.exists(precalc_file), f"Precalculated file for {agg_name} not found"

  incremental_dfs = [pd.read_parquet(file) for file in output_files]
  combined_df = pd.concat(incremental_dfs).drop_duplicates().reset_index(drop=True)
  
  precalc_df = pd.read_parquet(precalc_file)

  compare_dataframes(combined_df, precalc_df)

if __name__ == "__main__":
  pytest.main([__file__])