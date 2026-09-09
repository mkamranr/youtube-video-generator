"""Who owns which channels and jobs.

Channels carry `owner` (a user id) and every job inherits the owner of the channel it was
created for. Users see only their own records; admins see everything (services/auth.py
`can_access`). This module handles the two moments ownership has to be repaired:

  * migration - records written before ownership existed (or the legacy first-start seed
    channel, created before any account) have an empty owner: they go to the oldest admin.
    Idempotent: only empty owners are touched, so it runs at every startup and once more
    right after the first admin is created;
  * user deletion - the deleted account's channels and jobs move to the acting admin.
"""
from services import channels, users


def oldest_admin_id() -> str:
    admins = [u for u in users.list_users() if u.get("role") == "admin"]
    if not admins:
        return ""
    return min(admins, key=lambda u: u.get("created", 0))["id"]


def migrate_unowned(jobs) -> dict:
    """Assign every channel and job without an owner. No-op until an admin exists.
    jobs: iterable of pipeline.Job (the manager's live objects; saved when changed)."""
    admin = oldest_admin_id()
    if not admin:
        return {"channels": 0, "jobs": 0, "skipped": True}
    n_ch = channels.assign_unowned(admin)
    owners = {c["id"]: c.get("owner", "") for c in channels.list_channels()}
    n_jobs = 0
    for j in list(jobs):
        if not j.owner:
            j.owner = owners.get(j.channel_id) or admin
            j.save()
            n_jobs += 1
    return {"channels": n_ch, "jobs": n_jobs, "skipped": False}


def reassign(old_uid: str, new_uid: str, jobs) -> dict:
    """Move every channel and job of `old_uid` to `new_uid`."""
    n_ch = channels.reassign_owner(old_uid, new_uid)
    n_jobs = 0
    for j in list(jobs):
        if j.owner == old_uid:
            j.owner = new_uid
            j.save()
            n_jobs += 1
    return {"channels": n_ch, "jobs": n_jobs}
