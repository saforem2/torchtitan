# Perlmutter as a debug/verification host

Polaris went down mid-session on 2026-08-24; Perlmutter covered the
CPU-side verification work. Notes so the next switch is quicker.

## Why it works as a stand-in

`module load pytorch/2.13.0` matches the torch version in the Polaris
production venv (`2.13.0+cu129` there, `2.13.0+cu130` here), so
tensor-semantics reproductions are like-for-like rather than
approximate. That is enough for shape/layout bugs, dataloader unit
bugs, and anything else that is pure tensor math.

It is NOT a stand-in for the training chain: no torchtitan env, no
blendcorpus corpora, and the checkpoints live on Polaris `/eagle`
(step-5600 alone is 234 GB).

## Git pushes: use SSH, not HTTPS

A fresh clone defaults to an HTTPS remote, and pushing fails with

    remote: Invalid username or token. Password authentication is not
    supported for Git operations.

The fix is NOT to add a token -- do not put one on a shared
filesystem. Pushes already work because **the SSH agent is forwarded**
from the laptop, so Perlmutter borrows the local keys:

```console
$ ssh -T git@github.com
Hi saforem2! You've successfully authenticated, ...
```

So just point the remote at SSH:

```bash
git remote set-url origin git@github.com:saforem2/torchtitan.git
```

Confirm with `git push --dry-run origin HEAD:<branch>`: getting a
"fetch first" / non-fast-forward hint means auth SUCCEEDED and only the
ref is behind. An auth failure looks different (the token message
above).

Caveat: this depends on agent forwarding being live for the session. If
`ssh-add -l` on Perlmutter returns "no identities" the forward is not
up, and a push will fail no matter what the remote URL says -- reconnect
rather than adding credentials on the cluster.

## Paths

```
repo     /pscratch/sd/f/foremans/projects/saforem2/torchtitan
ezpz     /pscratch/sd/f/foremans/projects/saforem2/ezpz   (v0.21.14, old)
```

## Running the layout regression test

```bash
module load pytorch/2.13.0
python3 torchtitan/experiments/ezpz/tests/test_attn_unflatten.py
```

No distributed stack needed -- the test extracts and execs the shipped
unflatten lines rather than importing the module (importing pulls in
`spmd_types`, which is not present on a login node).
