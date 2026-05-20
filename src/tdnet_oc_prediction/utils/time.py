from datetime import datetime, timezone

def run_id(prefix='run'): return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
