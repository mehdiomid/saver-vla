# SAVER-VLA

**Feedback-driven agentic failure recovery for vision-language-action (VLA) manipulation.**

A VLA policy such as SmolVLA does nominal manipulation well, but when a grasp slips or
an object falls, it keeps executing actions that no longer apply. SAVER-VLA is a
supervisor that sits next to the policy. It watches each step, and when something goes
wrong it pauses the policy, asks an agent which recovery skill to run, executes that
skill safely, checks the result, and hands control back to the policy when the robot
is ready to try again. The policy is never retrained.

Project page: https://mehdiomid.github.io/saver-vla/

---

## How it works

```
             every step (20 Hz)                              only after a failure
 ┌────────┐  action  ┌────────────┐  state  ┌──────────┐  event  ┌─────────────┐  observation  ┌───────┐
 │ Policy │ ───────► │   Robot    │ ──────► │ Verifier │ ──────► │ Coordinator │ ────────────► │ Agent │
 │(VLA)   │          │ (sim/real) │         └──────────┘         │  (rules,    │ ◄──────────── │ (LLM) │
 └────────┘          └────────────┘                              │   budgets)  │  skill+reason └───────┘
      ▲                    ▲                                     └─────────────┘
      │ resume: bounded    │ bounded skill commands                 │
      │ grasp trial        └──────────── Skills ◄────────────────────┘
      └───────────────────────────────────────── outcome (REACHED / FAILED / VERIFIED) feeds back
```

1. **Detect.** The verifier checks grasp and transport every step and raises an event
   when the object stops moving with the gripper.
2. **Decide.** Only then is the agent called. It sees a JSON observation (geometry,
   grasp state, previous skill outcomes, and which skills are available or blocked) and
   picks one skill. It never outputs motor commands.
3. **Act.** The coordinator validates the request and runs a bounded skill: move to
   pregrasp, approach, shift the grasp pose, reposition, or reassess.
4. **Verify.** Each skill reports `REACHED` or `FAILED`. Returning control to the
   policy starts a bounded grasp trial that is reported `VERIFIED` or `NOT_VERIFIED`.
5. **Replan.** Every outcome goes back to the agent until the grasp is verified, the
   agent aborts, or the decision budget runs out.

---

## What's in the package

| Module | Role | Robot-specific? |
|---|---|---|
| `saver_vla/supervisor.py` | The closed loop above: switches between policy and skills, calls the agent at boundaries, runs grasp trials | No |
| `saver_vla/verifier.py` | Transport verifier (failure detection) and grasp tracker (trial verification) | No |
| `saver_vla/skills/` | Bounded recovery skills (pregrasp, approach, adjust grasp pose, 2D reposition, reassess) | Parameters only |
| `saver_vla/agents/` | Scripted baseline, GPT agent (final Phase 6.5.1 prompt), replay agent, custom planners | No |
| `saver_vla/coordinator.py` | Which actions are allowed right now, attempt limits, distance gates, rejection reasons | No |
| `saver_vla/faults.py` | Controlled fault injection for reliability testing | Some faults are sim-only |
| `saver_vla/robots/` | Robot profiles (Panda; SO-101 placeholder) and the backend interface | **Yes** |
| `saver_vla/robots/toy.py` | Tiny kinematic test world (seconds, no GPU) | Test only |
| `saver_vla/sim/libero.py` | LIBERO + SmolVLA backend built on LeRobot v0.6.1's public APIs | **Yes** |
| `saver_vla/runner.py` | Runs one episode and writes its logs | No |
| `saver_vla/sweep.py`, `metrics.py` | Many episodes, success rates with confidence intervals, reports | No |
| `saver_vla/config.py` | One typed YAML config per experiment | No |

Everything robot-specific lives in `robots/` and `sim/`. Moving to a new robot means adding
an adapter there; the recovery method itself does not change.

---

## Install

**Anywhere (unit tests and the toy world):**

```bash
pip install -e ".[llm,dev]"
pytest
```

**With LIBERO and SmolVLA (needs an NVIDIA GPU, e.g. Colab):** open
`notebooks/colab_quickstart.ipynb`. It installs LeRobot v0.6.1 with
`pip install -e ".[libero,smolvla]"`, pins `mujoco==3.3.2`, and then installs this package.
LeRobot is used **unmodified**.

---

## Quick start

```bash
# 1. Toy world, a few seconds, no GPU
saver run configs/toy_demo.yaml

# 2. The Phase 6.5.1 experiment in LIBERO (needs OPENAI_API_KEY)
saver run configs/phase651.yaml

# 3. Same experiment with any field changed from the command line
saver run configs/phase651.yaml --set recovery.model=gpt-5.2 --set seed=1001

# 4. Replay a previous run's decisions exactly, without API calls
saver run configs/phase651.yaml --set recovery.agent=replay \
                                --set recovery.replay_log=runs/phase651/<run>/events.jsonl

# 5. A reliability study, then its report
saver sweep configs/sweep_reliability.yaml --dry-run
saver sweep configs/sweep_reliability.yaml
saver report runs/reliability_v1
```

The replay agent also reads the notebook-era `agent_events.jsonl` files, so the original
GPT-5.5 decisions from Phase 6.5.1 can be replayed in the new package (see the Colab notebook).

---

## Configuration

One YAML file per experiment replaces the notebook's ~60 `SAVER_*` environment variables.
Every default equals the Phase 6.5.1 setting, so an empty file reproduces that experiment.

| Field | Default | Meaning |
|---|---|---|
| `env.backend` | `libero` | `libero` or `toy` |
| `env.suite`, `env.task_id` | `libero_object`, `0` | LIBERO task (soup can to basket) |
| `env.episode_length` | `850` | step limit |
| `seed`, `episode_index` | `1000`, `0` | seed (as in `lerobot-eval`) and LIBERO initial state |
| `faults.transport` | `gripper_release` | `none`, `forced_separation`, `gripper_release`, `velocity_impulse` |
| `faults.approach_bias_x_m` | `0.055` | hidden calibration bias on the centered approach |
| `faults.trial_impairment_steps` | `0` | force the gripper open during the first grasp trial |
| `recovery.agent` | `llm` | `none`, `queue_reset`, `scripted`, `llm`, `replay` |
| `recovery.model` | `gpt-5.5` | LLM model name |
| `recovery.blind` | `true` | hide the fault's identity from the agent |
| `recovery.capability_aware` | `true` | show available/blocked skills; replan on rejection |
| `recovery.skills` | all | skills the agent may use |
| `recovery.max_decisions`, `max_trials`, `trial_steps` | `8`, `2`, `95` | budgets |

How the test phases map onto configurations:

| Test phase                     | Configuration |
|--------------------------------|---|
| 3.x policy alone / queue reset | `recovery.agent: none` / `queue_reset` |
| 5 scripted vs. agentic         | `recovery.agent: scripted` / `llm`, `faults.transport: forced_separation` |
| 6.1 gripper release            | `faults.transport: gripper_release`, `faults.approach_bias_x_m: 0` |
| 6.3 impaired first trial       | `faults.trial_impairment_steps: 65` |
| 6.4a blind diagnosis           | `recovery.blind: true` |
| 6.4b approach bias             | `faults.approach_bias_x_m: 0.035` or `0.055` |
| 6.4c capability-aware          | `recovery.capability_aware: true` |
| 6.5 / 6.5.1 2D reposition      | `REPOSITION_TO_OBJECT_2D` in `recovery.skills` (6.5.1 height-aware version) |

---

## Outputs

Each episode folder contains:

- `config.yaml`: the exact configuration used.
- `events.jsonl`: every event in order: `FAULT`, `VERIFIER_EVENT`, `FAILURE`, `AGENT_CALL`,
  `DECISION` (with the full observation, the request and what was executed), `SKILL_REJECTED`,
  `SKILL_OUTCOME`, `POLICY_TRIAL_START`/`OUTCOME`, `RECOVERY_ABORTED`, `EPISODE_END`.
- `result.json`: the episode summary.

Every episode is labelled so task success is never credited to the agent by default:

| Outcome | Meaning |
|---|---|
| `agent_recovery` | a grasp trial supervised by the agent was verified, and the task succeeded |
| `policy_after_abort` | the task succeeded only after the agent ended its intervention |
| `resumed_success` | success after control returned without a verified trial |
| `policy_only_success` | success with no recovery involved |
| `nominal_success` | no fault was injected |
| `failure` | the task was not completed |

A sweep adds `summary.csv` (one row per episode), `summary.json` and `report.md` (per
condition: task success, detection rate, agent recovery rate, all with 95% Wilson
confidence intervals; mean decisions; detection latency).

---

## Extending

**A new skill.** Subclass `saver_vla.skills.base.Skill` (implement `start`, `update`,
`status`), add it to `ACTION_MENU` and `make_skill` in `saver_vla/skills/__init__.py`, give
it parameters in `SkillConfig`, and describe it in the agent prompt.

**A new agent.** Subclass `saver_vla.agents.Agent` and implement `_decide(observation)`,
or wrap a function with `CallableAgent`.

**Adding the SO-101 (planned).**
1. `robots/SO101Profile.to_action`: convert the skills' Cartesian commands into joint
   targets with inverse kinematics.
2. `robots/so101.py`: a backend that reads joint and gripper state from LeRobot's SO-101
   driver, runs the policy, and sends actions.
3. A state estimator for the object position from cameras, since there is no simulator
   state on hardware.
4. An SO-101 skill profile (reach, speeds, workspace limits) and safety checks.
Gripper release, trial impairment and approach bias faults only modify commands, so
they can be used on the real arm for supervised reliability tests.
