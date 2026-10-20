# Cloud orchestration

An on-demand worker VM for Alpha Titans: footage goes to private cloud
storage, a durable queue records the job, a worker is started only while
there is work, runs the **existing** pipeline unchanged, saves the results
back to storage, and powers itself off. The browser downloads them.

**Live since 17 Oct 2026** on project `alphatitans-cloud-1017`, with Cloud
Storage in place of Drive (a personal account has no Shared Drives); see the
deployment guide.

| | |
|---|---|
| [A. Repository findings](01-findings.md) | what exists, and where the brief and the code disagree |
| [B. Architecture decision record](02-adr.md) | the alternatives, the choice, diagrams, lifecycle, safeguards |
| [D. Deployment guide](03-deployment.md) | what is running, how to reproduce it, smoke test as run, hardening still to do |
| [E. Test report](04-test-report.md) | what ran, what passed, what was not and cannot be tested here |
| [F. Cost and lifecycle](05-cost.md) | what bills when, the batching model, missing prices |
| [G. Change report](06-change-report.md) | every file touched and why; remaining blockers |
| [NVENC proposal](07-nvenc-proposal.md) | **not implemented**: needs approval to touch the protected engine |

Code: `server/cloud/`. Tests: `tests/cloud/`. Deploy files: `deploy/cloud/`.
Page: `frontend/cloud.html` and `cloud.js`, served at `/cloud`.

Off unless `AT_CLOUD_ENABLED=1`. With it off, the app is exactly as before.
The control-plane VM runs `cloud.control_app`, which serves the cloud page
and its API and nothing else.
