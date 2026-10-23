# HuD04 v8: anonymous Google Drive data source

The approved public share is:

https://drive.google.com/file/d/18JsXNF4GDXUFlmlNBZ85zMT847Ks74DB/view?usp=sharing

Use the existing bootstrap with this URL:

```sh
gm-run 1021php/flux/git_smoke_v8.py --data-url 'https://drive.google.com/file/d/18JsXNF4GDXUFlmlNBZ85zMT847Ks74DB/view?usp=sharing'
```

This command is for a separately approved Flux GPU task. Preparing this code
does not authorize or start another paid task. Do not use the command to restart
an exhausted single-attempt task without fresh approval.

## Download and validation behavior

- Uses only the Python standard library, anonymously; no API key or browser token.
- Accepts only this file ID on the exact HTTPS Drive/download hosts and paths.
- Follows the normal public GET download confirmation, not login forms.
- Rejects other hosts, other IDs, credentials in URLs, unexpected ports, malformed
  confirmation forms, HTML loops, oversized downloads and truncated responses.
- Retains the 180-second total download deadline (including confirmation and
  wrapper normalization), 512 MiB transfer cap and no automatic retries.
- Supports the W&B export wrapper by reading only one approved inner archive,
  without extracting the rest of the wrapper tree.
- Requires the original 218231385-byte inner ZIP with SHA256
  `99c8225859128d27ec94dcc9ea7176e1257257926ae19900be40bebe4208d4e4`.
- Then verifies all 56 motion files against the embedded validation manifest.
- Keeps the 32-environment, two-PPO-iteration setup check and 900-second worker
  limit unchanged. Platform startup time is outside that worker limit.
- Logs diagnostic events without data URLs, confirmation tokens or credentials.

The Drive share currently contains the 218309932-byte W&B export wrapper, not
the original ZIP as its top-level file. Replacing the URL alone in an older
bootstrap is insufficient.

## Evidence and limits

The updated downloader passed a real anonymous CPU-sandbox download, inner SHA
check and all 56 manifest checks, plus 22 new CPU tests and 52 regression tests.
CPU-sandbox transfer speed is NOT evidence of Flux-worker throughput. In
particular, a previous GitHub Release path was fast in the sandbox but too slow
on Flux. No new GPU execution or simulator/PPO success is asserted here.

Run the focused CPU suite from `flux/tests`:

```sh
python -m unittest test_git_smoke test_download_diagnostics test_local_archive test_github_release_source test_google_drive_source -v
```