"""Private, optional cloud evidence archive. Never authoritative for promotion."""

import os
import httpx
from services.core import store


def mirror(campaign_id):
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        store.event(
            campaign_id, "archive", "Evidence saved locally. Supabase archive is not configured.", "data"
        )
        return
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    try:
        snapshot = store.get(campaign_id)
        with httpx.Client(timeout=60, headers=headers) as client:
            for a in snapshot.get("artifacts", []):
                path = store.ARTIFACTS / campaign_id / a["path"]
                r = client.post(
                    f"{url}/storage/v1/object/forge-artifacts/{campaign_id}/{a['path']}",
                    content=path.read_bytes(),
                    headers={"Content-Type": "application/octet-stream", "x-upsert": "false"},
                )
                if r.status_code not in (200, 201):
                    body = r.json()
                    if (
                        body.get("error") != "Duplicate"
                        and body.get("message") != "The resource already exists"
                    ):
                        r.raise_for_status()
            r = client.post(
                f"{url}/rest/v1/forge_campaigns",
                json={"id": campaign_id, "snapshot": snapshot},
                headers={"Prefer": "resolution=merge-duplicates"},
            )
            r.raise_for_status()
        store.event(campaign_id, "archive", "Evidence archived in private Supabase storage.", "data")
    except httpx.HTTPError as e:
        store.event(
            campaign_id,
            "archive_error",
            f"Cloud archive failed ({type(e).__name__}). Local evidence is preserved.",
            "data",
        )
