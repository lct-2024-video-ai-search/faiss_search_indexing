import os
from dotenv import load_dotenv
from clickhouse_driver import Client
import logging

load_dotenv()

host = os.getenv('CLICKHOUSE_HOST', 'localhost')
port = int(os.getenv('CLICKHOUSE_PORT', '9000'))
database = os.getenv('CLICKHOUSE_DATABASE', 'default')

client = Client(host=host, port=port)

def get_videos():
    query = '''
        SELECT 
            idx
            , audio_description
            , video_description
            , user_description
        FROM 
            VideoIndex
        '''

    result, columns = client.execute(query, with_column_types=True)
    logging.info(f"Получено видео: {len(result)}")
    return result, columns

