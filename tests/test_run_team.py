"""
Smoke test for run_team.py
"""
import os
import scripts.run_team as run_team

def test_run():
    config_path = "configs/team.yaml"
    endpoint = os.getenv("PROJECT_ENDPOINT", "test_endpoint")
    team_config = run_team.load_team_config(config_path)
    assert 'leader' in team_config
    assert 'workers' in team_config
