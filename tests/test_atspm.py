import pytest
import pandas as pd
import os
import shutil
from src.atspm import SignalDataProcessor
import duckdb
import numpy

# Define the parameters for testing
TEST_PARAMS = {
  'raw_data': duckdb.query("select * from 'tests/hires_test_data.parquet'").df(),
  'detector_config': duckdb.query("select * from 'tests/configs_test_data.parquet'").df(),
  'bin_size': 15,
  'output_dir': 'tests/test_output',
  'output_to_separate_folders': False,
  'output_format': 'parquet',
  'output_file_prefix': 'test_',
  'remove_incomplete': False,
  #'unmatched_event_settings': {
  #    'df_or_path': 'test_folder/unmatched.parquet', # for timeline, arrival_on_green & yellow_red
  #    'split_fail_df_or_path': 'test_folder/sf_unmatched.parquet', # just for split_failures
  #    'max_days_old': 14}, # remove unmatched events older than 14 days
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
      {'name': 'split_failures', 'params': {'red_time': 5, 'red_occupancy_threshold': 0.80, 'green_occupancy_threshold': 0.80, 'by_approach': True, 'by_cycle': False}},
      {'name': 'splits', 'params': {}},
      {'name': 'terminations', 'params': {}},
      {'name': 'yellow_red', 'params': {'latency_offset_seconds': 1.5, 'min_red_offset': -8}},
      {'name': 'timeline', 'params': {'min_duration': 0.2, 'cushion_time':60}}, # events shorter than 0.2 seconds are removed. coord pattern change events assigned duration of 60s (for visualization)
  ]
}

# Define aggregations that can be run incrementally
INCREMENTAL_AGGREGATIONS = [agg for agg in TEST_PARAMS['aggregations'] if agg['name'] not in ['unique_ped', 'full_ped', 'yellow_red']]

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

def round_specific_columns(df, columns_to_round, tenths=2):
  """Round specific columns in a dataframe to the nearest multiple of tenths."""
  for col in columns_to_round:
      if col in df.columns:
          df[col] = (df[col] / (0.1 * tenths)).round() * (0.1 * tenths)
  return df

def compare_dataframes_with_tolerance(df1, df2, tolerance):
  """
  Compare two dataframes, ignoring row order, applying rounding to specific columns,
  and allowing for a percentage of different datapoints.
  """
  # Columns to round (adjust as needed)
  columns_to_round = ['Green_Occupancy', 'Red_Occupancy']

  # Apply rounding
  df1 = round_specific_columns(df1, columns_to_round)
  df2 = round_specific_columns(df2, columns_to_round)

  # Sort dataframes
  df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
  df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

  # Compare dataframes
  comparison = df1_sorted.compare(df2_sorted)
  
  # Calculate the percentage of different datapoints
  total_datapoints = df1.size
  different_datapoints = comparison.size
  difference_percentage = different_datapoints / total_datapoints

  # Check if the difference is within the tolerance
  assert difference_percentage <= tolerance, f"Dataframes differ by {difference_percentage:.2%}, which is more than the allowed {tolerance:.2%}"



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
  data = duckdb.query("select * from 'tests/hires_test_data.parquet'").df()

  chunks = {
      '1_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 15:00:00' and timestamp < '2024-05-13 15:15:00'").df(),
      '2_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 15:15:00' and timestamp < '2024-05-13 15:30:00'").df(),
      '3_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 15:30:00' and timestamp < '2024-05-13 15:45:00'").df(),
      '4_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 15:45:00' and timestamp < '2024-05-13 16:00:00'").df(),
      '5_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 16:00:00' and timestamp < '2024-05-13 16:15:00'").df(),
      '6_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 16:15:00' and timestamp < '2024-05-13 16:30:00'").df(),
      '7_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 16:30:00' and timestamp < '2024-05-13 16:45:00'").df(),
      '8_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 16:45:00' and timestamp < '2024-05-13 17:00:00'").df(),
      '9_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 17:00:00' and timestamp < '2024-05-13 17:15:00'").df(),
      '10_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 17:15:00' and timestamp < '2024-05-13 17:30:00'").df(),
      '11_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 17:30:00' and timestamp < '2024-05-13 17:45:00'").df(),
      '12_chunk': duckdb.sql("select * from data where timestamp >= '2024-05-13 17:45:00' and timestamp < '2024-05-13 18:00:00'").df(),
  }

  output_dir = 'tests/test_incremental_output'
  os.makedirs(output_dir, exist_ok=True)

  for i, chunk in chunks.items():
      #if i != '1_chunk':
      #   continue
      params = TEST_PARAMS.copy()
      params.update({
          'raw_data': chunk,
          'output_dir': f'{output_dir}',
          'output_file_prefix': f"{i}_",
          'unmatched_event_settings': {
            'df_or_path': f"{output_dir}/unmatched.parquet", # for timeline, arrival_on_green & yellow_red
            'split_fail_df_or_path': f"{output_dir}/sf_unmatched.parquet", # just for split_failures
            'max_days_old': 14}, # remove unmatched events older than 14 days
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
  output_files = [os.path.join(incremental_processor_output, f"{i}_chunk_{agg_name}.parquet") for i in range(1, 13)]
  precalc_file = f"tests/precalculated/{agg_name}.parquet"

  for file in output_files:
      assert os.path.exists(file), f"Incremental output file {file} not found"
  assert os.path.exists(precalc_file), f"Precalculated file for {agg_name} not found"

  incremental_dfs = [pd.read_parquet(file) for file in output_files]
  combined_df = pd.concat(incremental_dfs).drop_duplicates().reset_index(drop=True)
  
  precalc_df = pd.read_parquet(precalc_file)

  # due to how split_failures imputes missing actuations there are some differences in incremental runs
  if agg_name == 'split_failures':
      compare_dataframes_with_tolerance(combined_df, precalc_df, tolerance=0.04)
  else:
      compare_dataframes(combined_df, precalc_df)

if __name__ == "__main__":
  pytest.main([__file__])