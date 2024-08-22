import os
from jinja2 import Environment, FileSystemLoader
from .utils import undo_rolling_sum

def render_query(query_name, **kwargs):
    # add from_table = 'raw_data' to the kwargs dictionary
    #kwargs['from_table'] = 'raw_data'

    # Get the directory that contains the SQL templates
    template_dir = os.path.join(os.path.dirname(__file__), 'queries')
    # Create a Jinja2 environment with the FileSystemLoader
    env = Environment(loader=FileSystemLoader(template_dir))
    # Get the template by name
    template = env.get_template(f"{query_name}.sql")
    # Render the template with the provided keyword arguments
    return template.render(**kwargs)

def aggregate_data(conn, aggregation_name, to_sql, **kwargs):
    query = render_query(aggregation_name, **kwargs)

    # Option to remove incomplete data (natural join with DeviceId, TimeStamp columns)
    if aggregation_name not in ['has_data', 'unmatched_events', 'timeline', 'split_failures'] and kwargs['remove_incomplete']:
        # Add natural join with has_data table
        query = f"SELECT * FROM ({query}) main_query NATURAL JOIN has_data "

    # Split failures are different to allow for saving incomplete cycle data
    if aggregation_name == 'split_failures':
        if kwargs['remove_incomplete']:
            query += " CREATE TABLE split_failures AS SELECT * FROM sf_final NATURAL JOIN has_data; "
        else:
            query += " CREATE TABLE split_failures AS SELECT * FROM sf_final; "
    else:
        query = f"CREATE TABLE {aggregation_name} AS {query}; "

    # For timeline aggregation, get unmatched rows (EndTime is null) and put them into table 'unmatched'
    if aggregation_name == 'timeline':
        # Insert unmatched rows into unmatched_events table if unmatched_event_settings is provided
        query += f""" CREATE TABLE unmatched_events AS
            SELECT StartTime AS TimeStamp, DeviceId, EventId, Parameter
            FROM timeline WHERE EndTime IS NULL; """
        # And drop unmatched rows from timeline table
        query += f" DELETE FROM timeline WHERE EndTime IS NULL OR Duration < {kwargs['min_duration']}; "
        # Drop EventId and Parameter from timeline table
        query += " ALTER TABLE timeline DROP COLUMN EventId; "
        query += " ALTER TABLE timeline DROP COLUMN Parameter; "

    try:
        if to_sql: # return sql as string
            return query
        # Otherwise, execute the query
        conn.query(query)

        # If agg is full_ped, check if return_volume is True
        if aggregation_name == 'full_ped':
            if kwargs['return_volumes']:
                # Create updated table with the volume data
                ped_data = conn.query("SELECT * FROM full_ped").df()
                #print('Dtypes of full_ped before adding Estimated_Volumes:\n', ped_data.dtypes)
                ped_data['Estimated_Volumes'] = ped_data.groupby(['DeviceId', 'Phase'])['Estimated_Hourly'].transform(undo_rolling_sum)
                #print('Dtypes of full_ped after adding Estimated_Volumes:\n', ped_data.dtypes)
                sql = """
                    CREATE OR REPLACE TABLE full_ped AS
                    SELECT * EXCLUDE (Estimated_Hourly)
                    FROM ped_data
                    WHERE PedServices >0 OR PedActuation >0 OR Unique_Actuations >0 OR Estimated_Volumes
                    """
                conn.query(sql)
        return None
    except Exception as e:
        print('Error when executing query for: ', aggregation_name)
        print(query)
        raise e
