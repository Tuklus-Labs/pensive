# Caddy: normalize Host on the pensive-shep block

> Historical record, written 2026-08-13. It describes the project as it stood then and is kept
> for provenance. Current behavior is documented in [README.md](README.md) and
> [daemon/README.md](daemon/README.md).

One line, operator-authorized (`/etc/*` is a Guardian-gated path, so this is Gary's
to apply, not mine). It lets the daemon go back to loopback-only defaults.

## Why

The daemon refuses unrecognized `Host` values to close DNS rebinding: an attacker
points a hostname they control at 127.0.0.1, the browser then treats the response
as same-origin so CORS never applies, and a GET is permitted to omit `Origin`
entirely, which is exactly the absence the Origin check has to allow for
non-browser clients. `Host` is the header the page cannot forge away.

The daemon's default allowlist is loopback only, which is the right default for
something that ships. But `pensive-shep` reverse-proxies to `127.0.0.1:5999`
with **no `header_up Host`**, and Caddy preserves the original Host, so
Shepherd's bridge from claude.ai arrives carrying `pensive.example.com`.
Today that is handled daemon-side by a systemd drop-in
(`80-allowed-hosts.conf`) that adds the public hostname to the allowlist.

That works, but it puts a public hostname in the daemon's trust set. Normalizing
at the proxy is better: the proxy is the trust boundary, and the daemon then
answers only to loopback no matter what is in front of it.

## The change

`/etc/caddy/Caddyfile`, in the `http://pensive.example.com` block, around
line 545:

```diff
                 reverse_proxy 127.0.0.1:5999 {
                     flush_interval -1
+                    header_up Host 127.0.0.1:5999
                 }
```

This is not a new pattern here. The `model-welfare` block on the same server
already does exactly this at line 580:

```
                header_up Host 127.0.0.1:6001
```

## Then

Once applied and `systemctl reload caddy` has run, the daemon can drop its
exception:

```bash
rm ~/.config/systemd/user/pensive-v3.service.d/80-allowed-hosts.conf
systemctl --user daemon-reload && systemctl --user restart pensive-v3
```

## Verified before proposing, not after

Against the live daemon, 2026-08-14:

| request | result |
|---|---|
| `Host: 127.0.0.1:5999` (what the change would send) | 200 |
| `Host: pensive.example.com` (the current path) | 200 |
| `Host: attacker.example` (the rebinding shape) | 403 |

So the daemon already accepts what Caddy would start sending, and the change
cannot break Shepherd's bridge. The order matters: apply the Caddy line first,
confirm the connector still answers from claude.ai, and only then remove the
daemon-side exception. Removing the exception first would sever the bridge until
the Caddy reload landed.

## What this does NOT fix

Nothing about the rebinding defense itself, which is already closed daemon-side
and stays closed either way. This is about where the normalization lives, and
about not carrying a public hostname in a daemon's trust set when the proxy in
front of it can strip the question entirely.
