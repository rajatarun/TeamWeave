import json
from pathlib import Path
import unittest

from src.orchestrator.schema_validate import validate_or_unwrap_output


class CreativeBriefSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema_path = Path("config/examples/schemas/creative_brief_v1.json")
        cls.schema = json.loads(schema_path.read_text(encoding="utf-8"))

    def test_accepts_objective_in_director_output(self):
        payload = {
            "goal": "Increase visibility in fintech",
            "objective": "Generate qualified inbound leads from LinkedIn",
            "audience": "Fintech CTOs",
            "channel": "linkedin",
            "acceptance_criteria": ["Concrete example included"],
            "risks": ["Overly broad positioning"],
            "success_metrics": ["10% engagement rate"],
        }
        self.assertEqual(validate_or_unwrap_output(payload, self.schema), payload)

    def test_maps_objective_to_goal_when_goal_missing(self):
        payload = {
            "objective": "Generate qualified inbound leads from LinkedIn",
            "audience": "Fintech CTOs",
            "channel": "linkedin",
            "acceptance_criteria": ["Concrete example included"],
            "risks": ["Overly broad positioning"],
            "success_metrics": ["10% engagement rate"],
        }
        validated = validate_or_unwrap_output(payload, self.schema)
        self.assertEqual(validated["goal"], payload["objective"])
        self.assertEqual(validated["objective"], payload["objective"])


if __name__ == "__main__":
    unittest.main()
