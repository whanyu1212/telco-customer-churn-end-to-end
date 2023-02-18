BQ_QUERY_TEMPLATE = """
SELECT *
FROM `{{project_id | sqlsafe}}.{{dataset_id | sqlsafe}}.{{table_id | sqlsafe}}`
"""
