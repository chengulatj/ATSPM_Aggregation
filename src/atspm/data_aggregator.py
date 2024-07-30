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
    if aggregation_name not in ['has_data', 'unmatched_events', 'timeline'] and kwargs['remove_incomplete']:
        # Add natural join with has_data table
        query = f"SELECT * FROM ({query}) main_query NATURAL JOIN has_data "
    query = f"CREATE OR REPLACE TABLE {aggregation_name} AS {query}; "

    # For timeline aggregation, get unmatched rows (EndTime is null) and put them into table 'unmatched'
    if aggregation_name == 'timeline':
        query += f"""CREATE OR REPLACE TABLE unmatched_events AS
                        SELECT StartTime AS TimeStamp, DeviceId, EventId, Parameter
                        FROM timeline
                        WHERE EndTime IS NULL
                        -- and less than n days old
                        AND StartTime > TIMESTAMP '{kwargs['min_timestamp']}' - INTERVAL '{kwargs['max_event_days']} days'
                        ; """
        # And drop unmatched rows from timeline table
        query += "DELETE FROM timeline WHERE EndTime IS NULL; "
        # Drop EventId and Parameter from timeline table
        query += "ALTER TABLE timeline DROP COLUMN EventId; "
        query += "ALTER TABLE timeline DROP COLUMN Parameter; "

    try:
        if to_sql: # return sql as string
            return query
        # Otherwise, execute the query
        conn.execute(query)

        # If agg is full_ped, check if return_volume is True
        if aggregation_name == 'full_ped':
            if kwargs['return_volumes']:
                # Create updated table with the volume data
                ped_data = conn.sql("SELECT * FROM full_ped").df()
                ped_data['Estimated_Volumes'] = ped_data.groupby(['DeviceId', 'Phase'])['Estimated_Hourly'].transform(undo_rolling_sum)
                sql = """
                    CREATE OR REPLACE TABLE full_ped AS
                    SELECT * EXCLUDE (Estimated_Hourly)
                    FROM ped_data
                    WHERE PedServices >0 OR PedActuation >0 OR Unique_Actuations >0 OR Estimated_Volumes
                    """
                conn.sql(sql)
        return None
    except Exception as e:
        print('Error when executing query for: ', aggregation_name)
        print(query)
        raise e
