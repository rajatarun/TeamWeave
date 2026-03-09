import json
import unittest
from pathlib import Path


class TeamConfigMetadataTests(unittest.TestCase):
    def test_all_example_team_agents_define_role_and_department_ids(self):
        team_files = Path("config/examples/teams").glob("*/v*/team.json")

        missing = []
        for team_file in team_files:
            data = json.loads(team_file.read_text(encoding="utf-8"))
            for agent in data.get("agents", []):
                if not agent.get("role_id") or not agent.get("department_id"):
                    missing.append(f"{team_file}:{agent.get('id')}")

        self.assertEqual(
            missing,
            [],
            f"Agents missing role_id and/or department_id: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
