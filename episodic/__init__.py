# episodic package

# Disable ChromaDB telemetry as early as possible
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Database safeguards are imported where needed to avoid loading ChromaDB early
# from . import db_safeguards
