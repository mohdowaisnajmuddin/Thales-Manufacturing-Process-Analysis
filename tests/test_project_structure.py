from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_required_project_paths_exist():
    required = [
        "data/raw/Thales_Group_Manufacturing.csv",
        "src/data_validation.py",
        "src/eda_analysis.py",
        "dashboard/app.py",
        "requirements.txt",
        "README.md",
    ]
    for relative in required:
        assert (ROOT / relative).exists(), f"Missing: {relative}"
