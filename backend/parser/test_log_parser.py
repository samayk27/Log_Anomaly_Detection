from backend.parser.log_parser import parse_logs
from backend.preprocessing.log_processor import process_logs


def test_loghub_hdfs_style_line_is_structured():
    logs = parse_logs('081109 203615 11 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_1 terminating')

    assert len(logs) == 1
    assert logs[0]['timestamp'] == '081109 203615'
    assert logs[0]['log_level'] == 'INFO'
    assert 'PacketResponder' in logs[0]['message']
    assert logs[0]['raw'].endswith('terminating')


def test_loghub_bgl_style_line_extracts_embedded_timestamp_and_level():
    logs = parse_logs('2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 1 service failed')

    assert len(logs) == 1
    assert logs[0]['timestamp'] == '2005.06.03 2005-06-03-15.42.50.363779'
    assert logs[0]['log_level'] == 'ERROR'
    assert logs[0]['dataset_label'] == 1


def test_json_and_csv_messages_are_processed_consistently():
    content = '\n'.join([
        '{"time":"2024-01-01T12:00:00Z","severity":"ERROR","logger":"api","event":"timeout"}',
        '2024-01-01 12:00:01,worker,WARN,"queue depth exceeded, retrying"',
    ])

    processed = process_logs(parse_logs(content))

    assert len(processed) == 2
    assert processed[0]['service'] == 'api'
    assert processed[0]['log_level'] == 'ERROR'
    assert processed[0]['cleaned_message'] == 'timeout'
    assert processed[1]['cleaned_message'] == 'queue depth exceeded retrying'


def test_apache_line_remains_supported():
    logs = parse_logs('127.0.0.1 - - [10/Oct/2000:13:55:36 -0700] "GET /health HTTP/1.1" 500 10')

    assert logs[0]['log_type'] == 'apache_nginx'
    assert logs[0]['status_code'] == '500'
    assert logs[0]['log_level'] == 'ERROR'