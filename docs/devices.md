# Device control

`device_controller` has one agent, `device_operator`. It takes an instruction,
looks up the devices DeviceWeave has registered, sends the command, reads the
resulting state, and reports what changed. The model category is `planning`:
a report that invents a device action is worse than a slower turn. The model
id comes from `config/model_map.yaml`. The team config does not name one.

The agent does not call DeviceWeave itself. The worker runs the `deviceweave`
tool before the turn and puts the result in `tool_results.deviceweave`. An
`error` key means the call did not happen. The agent is told to say that and
not to invent a device or a state.

## What DeviceWeave exposes

DeviceWeave is the HTTP API in [rajatarun/DeviceWeave](https://github.com/rajatarun/DeviceWeave).
It is not an MCP server, so it is not an AgentCore gateway target. The tool
calls the routes that API serves:

| Tool operation | HTTP | Body | What it is |
|---|---|---|---|
| list | `GET /devices` | — | Active registry. Each device has `id`, `name`, `device_type`, `capabilities`. IP addresses are not in this view. |
| status | `GET /devices/{device_id}` then `POST /execute` | `{"command": "status of <name>"}` | The registry record, then live state. DeviceWeave has no status route. Its intent parser maps the word "status" to the `get_status` action. |
| command | `POST /execute` | `{"command": "<instruction>"}` | One-shot control. No `session_id`, so this is not DeviceWeave's own conversational agent. |

The team's pre-tool runs list, then command, then a status read of each device
the command touched.

**Auth.** DeviceWeave's API Gateway HTTP API has no authorizer. Calls are
unauthenticated. `DeviceWeaveSecretArn` is optional. When it is set, the
worker reads the secret and sends `Authorization: Bearer`. The secret is
`{"token": "..."}`, `{"key": "..."}`, or the raw token. DeviceWeave does not
check that header today. Leave the parameter empty unless you have put a
bearer in front of the API. Do not put a key in the template or the team config.

Provider credentials (Kasa, SwitchBot, Govee, Ring, MyQ, Wyze) stay in
DeviceWeave's own secrets. TeamWeave does not read them.

## What you configure

Deploy DeviceWeave first. Its workflow names the stack
`deviceweave-${stage}`, and the default stage is `prod`, so the stack is
`deviceweave-prod`. The output TeamWeave reads is `ApiBaseUrl`.

TeamWeave's deploy resolves that output into the `DeviceWeaveApiBaseUrl`
parameter and sets `DEVICEWEAVE_URL` on the worker. A missing stack or a
stack with no `ApiBaseUrl` warns and passes an empty URL. The tool then
returns `error` and does not guess a host. It does not fail the TeamWeave
deploy.

To point a stack at DeviceWeave by hand:

```bash
aws cloudformation describe-stacks --stack-name deviceweave-prod \
  --query "Stacks[0].Outputs[?OutputKey=='ApiBaseUrl'].OutputValue" \
  --output text
```

Pass that value as `DeviceWeaveApiBaseUrl` on `sam deploy`. After the
TeamWeave stack exists, `scripts/stack_env.py` publishes it as
`TEAMWEAVE_DEVICEWEAVE_URL` from the `DeviceWeaveUrl` output.

Devices themselves are registered by DeviceWeave, not by this team. Run
DeviceWeave's `POST /ingest` (for example `{"provider": "kasa", "mode": "full"}`)
so `GET /devices` is not an empty catalog.

## Risky commands

These are not sent unless the request's `confirm` field is `yes`, `true`,
`1`, or `confirm`:

- unlock, including opening a garage door (`MyQGarageDoor`'s action is `open`)
- disable or disarm security, an alarm, a camera, a doorbell, a lock, or a sensor
- factory reset
- powering off a critical device

Critical device types, from DeviceWeave's adapters: `WyzeLock`,
`RingDoorbell`, `RingCamera`, `WyzeCamera`, `WyzeMotionSensor`,
`WyzeContactSensor`, `MyQGarageDoor`. A command that says lock, alarm,
security, camera, doorbell, garage, or sensor and asks to turn it off is
held even when the name does not match a registry row. Turning off a light,
a plug, or a fan is sent.

When a command is held, the tool does not call `POST /execute`. The result
has `executed: false` and one `needs_confirmation` entry (`device`,
`command`, `reason`). The agent's `actions_taken` stays empty.

## How to run it

`POST /team/task` returns 202 and a `run_id`. Poll `GET /team/task/{run_id}`.
The API is behind the SIWE authorizer. The body:

```bash
curl -sS -X POST "$TEAMWEAVE_API_BASE/team/task" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "team": "device_controller",
    "version": "v1",
    "request": {"instruction": "Turn on the office light"}
  }'
```

A risky command that should actually run:

```bash
-d '{
  "team": "device_controller",
  "version": "v1",
  "request": {
    "instruction": "Unlock the front door",
    "confirm": "yes"
  }
}'
```

The same `confirm` left off produces a proposed action in
`needs_confirmation` and does not change the device.

## Output

The step validates `device_action_v1`:

| Field | Meaning |
|---|---|
| `summary` | What changed, or that nothing ran |
| `actions_taken[]` | `device`, `command`, `result` for each command DeviceWeave executed |
| `current_state[]` | `device` and `state` read back afterwards |
| `assumptions` | What was assumed instead of asking |
| `errors` | Calls that failed, including a status read that did not come back |
| `needs_confirmation[]` | `device`, `command`, `reason` for a command that was held |

`current_state` is a list, one entry per device, so a scene that touches two
devices can report both.
