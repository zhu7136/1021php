# HuD04 v8: public Release data source (no storage mount)

The owner approved publishing the original v8 56-motion candidate ZIP as a
public GitHub Release asset. This is a simulation-only setup-check dataset,
not validated hardware deployment data or evidence of successful PPO training.

- Release tag: `hud04-data-v8-56-clips`
- Asset: `hud04-cpu-repair-v8-56-clips.zip`
- Exact bytes: `218231385`
- SHA256: `99c8225859128d27ec94dcc9ea7176e1257257926ae19900be40bebe4208d4e4`
- Public URL:
  `https://github.com/zhu7136/1021php/releases/download/hud04-data-v8-56-clips/hud04-cpu-repair-v8-56-clips.zip`

Once publication and the complete download have been independently verified,
the Flux start command can be:

```sh
gm-run 1021php/flux/git_smoke_v8.py --data-url https://github.com/zhu7136/1021php/releases/download/hud04-data-v8-56-clips/hud04-cpu-repair-v8-56-clips.zip
```

For this route, disable personal-storage mounting in the task configuration.
An empty `personalDataPath` was used by the earlier non-mounting tasks.
Use branch `aria/hud04-flux-teacher-20261023` and verify its deployed commit
before starting a separately approved trial.

The bootstrap sends no GitHub token to the training task. Only the exact
approved Release path and GitHub asset CDN hosts are accepted, in addition
to the existing W&B artifact GCS hosts. HTTPS is required and redirects are
checked. Do not put signed redirect URLs or credentials in Git.

The original SHA256, complete 56-motion batch, per-file manifest hashes,
180-second total download deadline and 900-second worker bound remain in
force. There is no download retry or partial-dataset fallback. A failed or
incomplete transfer exits without installing dependencies or entering PPO.
Setup-check settings remain 32 environments and 2 PPO iterations.

Sandbox download speed does not establish speed from the Flux GPU worker.
Creating this Release or editing code does not authorize paid training.