import os

# NEED TO ADD IN OPTION TO APPEND TO EXISTING FILE!!!!!!!!

def save_data(**kwargs):
    # Extract parameters from kwargs
    output_dir = kwargs['output_dir']
    output_to_separate_folders = kwargs['output_to_separate_folders']
    output_format = kwargs['output_format']
    # check if output_file_prefix is in kwargs, if not set to empty string
    if 'output_file_prefix' not in kwargs:
        prefix = ''
    else:
        prefix = kwargs['output_file_prefix']
    conn = kwargs['conn']
    if 'unmatched_events' in kwargs:
        unmatched_events_path = kwargs['unmatched_events']
    else:
        unmatched_events_path = None

    # Make output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all table names in the database
    table_names = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    # Iterate over all tables
    for table_name in table_names:
        table_name = table_name[0]
        # skip if table name is raw_data or detector_config
        if table_name in ['raw_data', 'detector_config']:
            continue
        # Save unmatched data to path given, if any
        if table_name == 'unmatched_events' and unmatched_events_path is not None:
            query = f"COPY (SELECT * FROM unmatched_events) TO '{unmatched_events_path}'"
            conn.execute(query)
            continue

        if output_to_separate_folders:
            final_path = f"{table_name}/{prefix}"
            # Create a directory for the table if it does not exist
            os.makedirs(f"{output_dir}/{table_name}", exist_ok=True)
        else:
            final_path = f"{prefix}{table_name}"

        # Query to select all data from the table
        query = f"COPY (SELECT * FROM {table_name} ORDER BY TimeStamp) TO '{output_dir}/{final_path}.{output_format}'"
        conn.execute(query)