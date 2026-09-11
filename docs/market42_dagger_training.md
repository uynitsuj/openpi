# Market42 DAgger training: action labels, controller tuning, and ABC

This guide includes the implemented offline Market42 converter and v12 DAgger
continuation recipe. Human review and explicit dataset splits are still required:
the tools do not automatically declare a recording suitable for imitation.
Do not feed relative-Gello Market42 episodes into the old Siemens converter.

For commands and manifest examples, start at [the runbook](#8-running-the-implemented-pipeline).

Audited 2026-09-11 against OpenPI `4747d45c59466bcf4b9c350ad544944bea81e352`,
Lab42 `656c57e698ee02c542878e60248ff4d782be75da`, and
[ABC `6bc6586`](https://github.com/amazon-far/abc/commit/6bc6586721cf0c409ccee80f675a28de9b9b2f5e)
(public main, committed 2026-07-02). Re-audit these contracts when the recorder,
controller, converter, or training configuration changes.

## Recommended starting point

- Keep complete raw rollouts, including failures, interventions, and outcomes.
- Collect future Siemens corrections with the original collection tuning, using
  Market42's existing `gravity_comp_profile: "v8dj_recorded"` preset on both YAMs.
  Its name is historical; it is not restricted to v8 checkpoints.
- Separate action authority, recording validity, and behavior quality. A human
  action is not automatically good; an autonomous action is not automatically bad.
- Convert follower-space commands, not raw passive-Gello joint positions, into
  training targets. Review handoffs and command timing before building chunks.
- Compare original-data-only training, original plus reviewed corrections, and
  an ABC-inspired mixture that also includes reviewed autonomous continuation.
  Treat mixed-gain inclusion and RTC changes as separate experimental variables.

## 1. Which controller settings are we matching?

The original Lab42
[`xdof/robot_configs/yam/base.yaml`](https://github.com/xdofai/lab42/blob/656c57e698ee02c542878e60248ff4d782be75da/xdof/robot_configs/yam/base.yaml)
already specifies the collection tuning. Market42's `v8dj_recorded` is an
explicit compatibility preset for these values, not a new v8-specific controller.

| Setting | Original collection / `v8dj_recorded` | Market42 `robots_realtime_dagger` |
| --- | --- | --- |
| `kp`, joints 1–6 plus gripper | `[80, 80, 80, 10, 10, 10, 20]` | `[80, 80, 80, 40, 15, 15, 20]` |
| `kd`, same order | `[5, 5, 5, 1.5, 1.5, 1.5, 0.5]` | Same |
| Gravity multiplier | `1.3` per modeled DOF | Same |
| Gravity-idle damping | Zero | Same |
| Coulomb compensation | Disabled | Same |
| Gripper effort limit in the audited station | `50` | Same |

These are the inspected **Market42 presets**, not a claim that every external
robots_realtime configuration uses these values. See
[`hardware.py`, profile gains](https://github.com/xdofai/lab42/blob/656c57e698ee02c542878e60248ff4d782be75da/packages/market42/src/market42/nodes/robots/hardware.py#L589-L594)
and [gravity/damping overrides](https://github.com/xdofai/lab42/blob/656c57e698ee02c542878e60248ff4d782be75da/packages/market42/src/market42/nodes/robots/hardware.py#L764-L777).

A stratified local audit of original `metadata.json` files from 18 episodes in
the [v12 manifest](../job_episodes_simple_d405_v12dj_recent.csv) found this same
collection tuning on both arms: first and last episodes in nine station/date
groups, stations `sz_34` and `sz_44`, September 2–9, 2026. This is a sample, not
a complete dataset audit. Audited manifest SHA-256:
`9a87261da0a3b28c1b890de1603bc7e0e0818470a44a93e02b2bfe11bcda333d`.

Do not confuse the collection config with today's i2rt factory defaults.
Market42's legacy `market42_default` delegates to the factory. At the audited
[i2rt revision, `yam_v1.yml`](https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/i2rt/robots/config/yam_v1.yml),
arm `kp/kd` match, but gravity factors are `[1.0, 1.1, 1.1, 1.2, 1.0, 1.0]`
and gravity-idle damping is `[0.1, 0.1, 0.1, 0.3, 0.05, 0.05]`.
The factory defaults Coulomb compensation to off even though it loads nonzero
coefficients. The recorded preset overrides gravity factors and idle damping;
switching back to the factory preset would therefore not preserve the full tuning.

Parameter agreement is not full dynamics agreement. The collection YAML points
to a legacy YAM model, while Market42 constructs its robot through the current
i2rt factory. Gravity-model inertias, end-effector mass, calibration, friction,
limits, gripper behavior, and command timing can still differ. Archive effective
runtime values and model/config hashes, not just the preset name. Do not copy
another station's motor zeros, gripper calibration, or safety limits.

### Audit of collected DAgger data (2026-09-11)

The corrections collected so far under `/nfs_exp/yiming/data/` were **not**
recorded with `v8dj_recorded`, per each episode's recorded `station_config`:

| Collection | Episodes kept | Lab42 commit | Recorded profile |
| --- | ---: | --- | --- |
| `20260910/` | 63 | `818ad01` (dirty) | field absent — predates `gravity_comp_profile`; factory behavior |
| `20260911/20260911/` | 31 | `39c739c` (dirty) | `market42_default` (both YAMs) |
| `20260911/20260911_updated_gain/` | 14 | `656c57e` (dirty) | `market42_default` (both YAMs) |

The i2rt submodule is `47fee5e7d` in all three (the revision audited above), so
`market42_default` resolves to the factory gravity model: arm `kp/kd` match
`v8dj_recorded` numerically, but gravity factors and idle damping differ.
The `updated_gain` rename is **not** visible in the recorded config or
submodules — every collection ran a dirty Lab42 tree. Numeric effective gains
are not archived in these episodes (only the profile name, and for 20260910
not even that). `cleanup_manifest.json` at `/nfs_exp/yiming/data/20260911/`
documents the kept/dropped sets; the on-disk directories are already
deduplicated.

**Empirical check (2026-09-11).** The collector's recollection was that the
0911 "normal" batch ran `robots_realtime_dagger` gains and `updated_gain` ran
`v8dj_recorded`. The recorded follower dynamics contradict this: over 8–10
episodes per group, per arm, both the median tracking error |command − state|
and the quasi-static (|vel| < 0.03 rad/s) signed residual are statistically
identical between the two 0911 groups on every joint (e.g. right j4 residual
+22.5 ± 6.9 vs +20.2 ± 9.9 mrad; left j4 +11.3 vs +11.2). The wrist-kp gap
between those two presets (40/15/15 vs 10/10/10) would scale the quasi-static
wrist residual ~2.5–4x; no shift of any size is present, nor any
gravity-model shift on j2–j4. Consistent with the recorded
`market42_default` on both — at the recorded revisions `hardware.py` applies
profile kp/gravity overrides only when the profile is not `market42_default` —
the intended gain update evidently did **not** take effect on the follower
controllers (or touched only a non-follower component, e.g. leader-side feel).
Treat both 0911 groups as one factory-gain regime in review manifests unless
station-side evidence emerges; do not label either batch `v8dj_recorded` or
`robots_realtime_dagger`.

### Does passive-Gello IK change the gravity compensation during intervention?

No. In the inspected
[control manager](https://github.com/xdofai/lab42/blob/656c57e698ee02c542878e60248ff4d782be75da/packages/market42/src/market42/policy_control/manager.py#L530),
takeover anchors the IK clutch to the measured follower pose and switches command
links. The same YAM node and motor controller remain active; it does not replace
their gains or gravity model. Both policy targets and IK targets pass through
the same YAM command relay. "Passive" refers to the unpowered Gello leaders;
the follower YAMs remain position-controlled, not freely backdrivable.

Relative IK solves a **kinematic** problem: the operator's end-effector motion
becomes a follower-space joint target without snapping to the Gello's absolute
joint pose. It does not cancel a gravity-model error or make different motor
gains dynamically equivalent. Distinguish three possible mismatches:

- Wrong IK geometry, joint signs, or tool frame: incorrect motion mapping.
- Wrong gravity model/gains: different tracking, effort, or contact response.
- Wrong handoff timing/anchor: a discontinuous target even with correct models.

Do not try to solve these by changing gravity compensation only during teleop.
Keep one verified controller profile throughout the rollout and log any explicit
changes. Physical verification requires supervised, arms-clear tests; this guide
does not authorize motion or controller changes.

## 2. Can data collected with different gains be mixed?

Potentially, yes; the source is not invalid merely because its gains differ.
But the same desired joint position can generate different motion and contact
forces under different controllers. ACT explicitly separates leader targets
from measured follower state because their difference matters to low-level
control. This is why replacing actions with measured future joints is not a
neutral cleanup. [ACT, Section IV](https://arxiv.org/html/2304.13705v1#S4)

Dynamics-randomization work deliberately varies gains and other dynamics, but
does so in an RL training setup. It does not establish that naively combining
our mixed-gain behavior-cloning recordings improves pi0.5.
[Peng et al., Sections IV–V](https://arxiv.org/html/1710.06537v2)

For this dataset, our recommendation is:

- Preserve older, stiffer-gain episodes with explicit provenance and review
  recovery quality. Do not discard a useful correction because the preceding
  policy motion failed, or accept all actions because the final episode succeeded.
- Review contact phases particularly carefully: grasping and bag flattening can
  depend on controller response. This is an engineering concern, not a measured
  mixed-training result for this task.
- Do not scale targets by a gain ratio or smooth them blindly to invent
  softer-controller demonstrations. Available positions do not uniquely identify
  the contact forces or counterfactual expert commands.
- Stratify evaluation by collection gains. For a fixed deployment profile,
  start without adding gain conditioning to the model; test its need separately
  if deployment must span several regimes.

Adding human-labeled recovery data to the original demonstrations is a
defensible baseline, as in [HG-DAgger](https://arxiv.org/html/1810.02890v2), but
it is not ABC's complete recipe or the only valid use of a rollout.

## 3. What ABC actually does

### Reported DAgger procedure

ABC records complete rollouts and continues the previous model for 75,000 steps
per round: **80% previous-round data, 10% new interventions, 10% new remainder**.
This outperformed whole-rollout or intervention-only training; the latter
exacerbated a cage/no-cage shortcut. This is task-specific evidence, not a Siemens
optimum. [Paper, Section 5.3](https://arxiv.org/html/2606.27375v1#S5.SS3)

Episodes start under policy control. Buttons switch to/from relative-SE(3)
passive-leader teleop with Mink IK. Hand-back uses recent intervention actions
as an RTC prefix.
[Paper, Appendix F](https://arxiv.org/html/2606.27375v1#A6)

Long prefixes can perpetuate failed motion; shorter prefixes trade smoothness
for visual responsiveness. [Paper, Appendix H.2](https://arxiv.org/html/2606.27375v1#A8.SS2)

### Verified public implementation

The pinned [README](https://github.com/amazon-far/abc/blob/6bc6586721cf0c409ccee80f675a28de9b9b2f5e/README.md#L14-L22)
describes a minimal DiT/conversion release and lists real deployment among work
still to be released. It is not an end-to-end public DAgger implementation.

| Component | Verified behavior | Important limit |
| --- | --- | --- |
| [MCAP export](https://github.com/amazon-far/abc/blob/6bc6586721cf0c409ccee80f675a28de9b9b2f5e/export_mcap.py#L119-L168) | Separate 14D state and action channels; MCAP `log_time`; causal floor sampling on a 30 Hz common-overlap grid; 28-column float64 output. | No authority/outcome fields. Missing scalar streams can become zeros; no freshness limit. Action-channel names do not establish how the unavailable recorder produced them. |
| [EpisodeDataset / MixtureDataset](https://github.com/amazon-far/abc/blob/6bc6586721cf0c409ccee80f675a28de9b9b2f5e/abc_minimal/train_loop.py#L154-L230) | State/images at chunk start; full future action chunk. Choose dataset source by weight, then a usable start uniformly within it. | No authority-aware chunk boundaries. Longer episodes supply more starts. |
| [Training config](https://github.com/amazon-far/abc/blob/6bc6586721cf0c409ccee80f675a28de9b9b2f5e/abc_minimal/config.py#L120-L160) | Configurable source mixture; shipped preset is 81.72% real bottles / 18.28% simulation. | This is not the paper's DAgger mixture or automatic round-to-round continuation. |
| [Flow objective](https://github.com/amazon-far/abc/blob/6bc6586721cf0c409ccee80f675a28de9b9b2f5e/abc_minimal/dit.py#L764-L812) | Train with clean/noise-perturbed prefix positions at diffusion time zero; loss only on the remaining positions. | The prefix loss mask is not a human-authority mask. |
| [RTC sampler](https://github.com/amazon-far/abc/blob/6bc6586721cf0c409ccee80f675a28de9b9b2f5e/abc_minimal/dit.py#L840-L876) | Clamp prefix values throughout denoising; use per-position diffusion times. | Depends on prefix-conditioned training, not just an inference flag. |
| [Simulation RTC manager](https://github.com/amazon-far/abc/blob/6bc6586721cf0c409ccee80f675a28de9b9b2f5e/abc_minimal/eval_policy.py#L530-L584) | Prefix comes from the tail of the current planned execution window; worker inference returns a chunk with prefix entries removed. | Planned commands, not measured joint history or a released real-robot intervention recorder. |

Implementation detail: `max_action_prefix=4` samples lengths 0–3 because
`torch.randint` excludes its upper bound. Prefix conditioning is disabled on
state-dropped samples. Do not describe this as sampling 0–4 inclusively.

The released code does not establish failed-episode retention rules, exact
intervention boundary treatment, handoff exclusion windows, or controller-gain
mixing behavior. Those must be designed and tested here. Neither the generic
ABC training command nor our ABC-format loader reproduces the paper's DAgger
experiment by itself.

## 4. Market42 action-source contract

Inspect actual channels and dimensions; node names are configurable. Typical
artifacts for the current bimanual station are:

| Artifact / topic | Meaning | Training use |
| --- | --- | --- |
| `left.mcap`, `right.mcap`: `/{side}-robot-state`, `/{side}-gripper-state` | Measured follower joints and gripper. | Observation/state, not a substitute for expert actions. |
| `action-left.mcap`, `action-right.mcap` | Raw passive-Gello positions. | Leader diagnostics. **Not follower-space targets under relative IK.** |
| `action-<ik_node>.mcap`: `/action-<ik_node>-left-robot-state`, corresponding right topic | Published follower-space IK targets, including gripper. | Human intent/provenance; compare against relay output. |
| `<policy_node>.mcap`: `/<policy_node>-left-robot-state`, corresponding right topic | Published policy targets. | Policy diagnostics; do not assume a published/shadow command was executed. |
| `left.mcap`, `right.mcap`: `/{side}-command-state` | Application-level relayed target; semantics depend on recording mode/version. | Candidate common action source after verification below. |
| `session_meta.json`: `dagger.segments` | `authority`, `started_at`, `ended_at` for policy/teleop intervals. | Authority annotation, not an actuator acknowledgment or quality label. |
| `eval_anno.json` | Operator episode outcome. | Review/evaluation metadata; absence is unknown, not failure or success. |

Sources: [recording contract](https://github.com/xdofai/lab42/blob/656c57e698ee02c542878e60248ff4d782be75da/packages/market42/docs/dagger-collection.md#episode-artifacts--upload),
[writer topic routing](https://github.com/xdofai/lab42/blob/656c57e698ee02c542878e60248ff4d782be75da/packages/market42/src/market42/core/writer.py#L649),
and [IK anchoring/output](https://github.com/xdofai/lab42/blob/656c57e698ee02c542878e60248ff4d782be75da/packages/market42/src/market42/nodes/controllers/passive_gello_ik.py#L789).

In the inspected fixed-rate/sync
[`YamRobotNode.step`](https://github.com/xdofai/lab42/blob/656c57e698ee02c542878e60248ff4d782be75da/packages/market42/src/market42/nodes/robots/hardware.py#L948),
`command-state` contains post-ramp `_cmd_current` at local step wall time. The
subscriber-driven async branch instead records upstream `jp` at source time;
during a ramp it need not equal the relayed target. Verify historical mode and
revision before trusting it. Neither is a direct motor-bus capture: downstream
limits or gripper force limiting may further modify the request.

Recommended target contract: verified follower-space relayed position requests,
with the same meaning across policy and teleop. Preserve upstream IK and policy
streams for auditing. For stable passthrough intervals, verify upstream and relay
agreement; do not silently substitute one source for missing data in another.
Treat ramps, stalls, uncertain authority, and source disagreements explicitly.

### Why the existing v12 converter is unsuitable unchanged

[`build_state_and_actions`](../scripts/yam_data/convert_xdof_mcap_job.py)
reads `action-left/right.mcap` and independently nearest-aligns it with follower
state to `timestamp.npy`. It does not resolve relative IK, relayed commands, or
DAgger authority. Market42 async recordings can also lack `timestamp.npy`.
The similarly named [ABC-layout exporter](../scripts/yam_data/export_abc_layout_job.py)
still reads those raw action files; selecting that layout does not fix semantics.

## 5. Conversion, chunking, and sampling requirements

The implemented [converter](../scripts/yam_data/convert_market42_dagger.py) and
[chunk dataset](../src/openpi/training/dagger_dataset.py) enforce the timestamp,
source-agreement, review, and full-chunk checks below. They preserve recorded
metadata, including unknown/missing fields; they cannot recover unrecorded
motor-bus behavior or certify controller calibration and behavior quality.

1. Archive raw data and create a versioned manifest with episode/round IDs,
   generating checkpoint, outcome, station, effective gains/gravity/friction,
   model/calibration hashes, command rates, handoff settings, RTC mode, and
   per-camera preprocessing. Unknown values remain unknown; do not infer gains
   from a checkpoint or relabel older episodes with today's default profile.
2. Decode each stream by its own timestamps. Normalize timestamp units, verify
   clocks/monotonicity, require both arms and required cameras, and build a
   common 30 Hz timeline. Prefer causal sample-and-hold for commands with a
   declared freshness bound. Reject gaps, missing channels, or out-of-support
   samples instead of zero filling or holding indefinitely. Validate action/obs
   latency with traces; do not introduce a one-step shift without evidence.
3. Derive authority from the segment timeline and cross-check commands around
   switches. Segment times are control-manager wall-clock events recorded after
   transitions, not per-arm actuator switch times. Mark ambiguous boundaries
   invalid using measured timing/ramp evidence, not an invented universal margin.
4. Store validity, authority, and review eligibility separately. Keep policy-only
   jitter, resets, unsafe contact, or stalls out of imitation targets unless an
   explicit experiment justifies them. Retain good corrections from failed
   episodes and reviewed useful autonomous continuation as distinct sources.
5. First support full 30-step chunks contained in a contiguous accepted interval.
   For an intervention chunk, every target must be valid, reviewed, and human
   controlled. An anchor-frame human flag alone is insufficient. Never join
   separated interventions into a fictitious continuous trajectory; exclude
   incomplete tails rather than silently padding them as expert behavior.
6. Keep source probabilities independent of raw frame counts. Report actual
   sampled shares, eligible frames/chunks, rejected reasons, and unique episodes
   per source/profile. An 80:10:10 sampler refers to training draws, not an
   episode-count split or a scalar loss multiplier.

Conservative full-chunk selection loses short interventions and near-boundary
recovery labels; report that loss. A later timestep-mask design must carry masks
through conversion, repacking, chunking, and loss reduction. Per-arm masks must
apply before averaging action dimensions. Also prevent invalid actions from
leaking into valid predictions as denoiser context; masking only their direct
loss does not accomplish that. This requires an explicit model/data design.

The current [v12 config](../src/openpi/training/config.py) and
[`YamInputs`](../src/openpi/policies/yam_policy.py) use absolute 14D actions,
driver order `[left joints 1–6, left gripper, right joints 1–6, right gripper]`,
with no delta-action conversion. Relative end-effector teleoperation does not
change that model action space. Preserve radians/gripper conventions, the
30-action horizon, and the exact prompt.

The [v12 conversion command](../sky/convert_siemens_simple_d405_v12dj_recent.yaml)
center-crops **all three cameras** before resizing to 224 square; v11 crops only
the top and pads wrists. Follow executable arguments/config, not the stale
padding comment in the v12 YAML. Do not copy ABC's letterboxing into v12 training
or serving. Keep the approved normalization transform and stats with the
checkpoint; either explicitly retain compatible stats for continuation or
version newly computed training-only stats and validate the resulting model.
Never silently use another dataset's stats to satisfy a name lookup.

## 6. Implemented chunk filtering and remaining model work

At the audited revision, [`Pi0.compute_loss`](../src/openpi/models/pi0.py)
returns per-timestep losses after averaging action dimensions.
[`train_step`](../scripts/train.py) averages the chunk, with optional scalar
sample weights only when RABC/online-RM weighting is enabled. The v12 repack does
not carry authority/validity, and a metadata column alone will not affect loss.
Existing RABC reward weighting is not DAgger authority masking.

The DAgger loader now resolves authority/validity/review **before** model
transforms: it only exposes full-horizon eligible chunks. Thus every target in
an intervention sample is human-controlled, with no invalid-context leakage or
changes to the existing loss. This deliberately does not implement partial-chunk
or per-arm timestep masking. Those remain separate model/data work.

OpenPI's current `sample_actions_rtc` uses inference-time VJP guidance; ordinary
training here does not implement ABC's per-position prefix-conditioned objective.
ABC's sampler therefore is not a drop-in improvement for an existing pi0.5
checkpoint. Training-time prefix augmentation is a separate model experiment.

Likewise, do not confuse past executed intervention actions with OpenPI's
future, not-yet-consumed action prefix. The current Market42
[`resume_from_observation`](https://github.com/xdofai/lab42/blob/656c57e698ee02c542878e60248ff4d782be75da/packages/market42/src/market42/agent/async_openpi_agent.py#L265)
resets stale plans and reseeds RTC from the hand-back observation; it is not a
reimplementation of ABC's recorded-human-history conditioning. Any adoption
needs explicit prefix/observation time alignment and tests for stale commands,
takeover/release, delayed inference, and failed-grasp responsiveness.

## 7. First training experiment and release gate

The converter, eligible-chunk sampler, and training entry point below implement
this first experiment. Complete the real-data review and release checks before
a production run; passing synthetic tests alone is not task-quality evidence.

Use the same initialization, optimizer budget, camera preprocessing, prompt,
normalization policy, and deployment settings across comparison arms:

| Experiment | Data | Purpose |
| --- | --- | --- |
| A | Original accepted demonstrations only | Continuation baseline. |
| B | Original data plus reviewed human recovery chunks | Test correction value; explicitly choose/log the mixture. |
| C | ABC-inspired 80:10:10 old / reviewed new intervention / reviewed new remainder | Test useful autonomous context with correction oversampling. Curation makes this an adaptation, not an exact reproduction. |

For each useful mixture, compare inclusion/exclusion of older stiffer-gain
corrections under a matched budget. Keep entire original episodes/collection
sessions in one split, including all clips derived from them. Audit source and
scene correlations so validation does not merely recognize the collection site.

Required offline checks: replay selected converted state/actions against the
original video and authority timeline; inspect takeover/release, contact,
gripper motion, and every controller profile; verify shapes/order/units, finite
values, timing/freshness, absence of gaps inside chunks, and no test leakage.
Synthetic fixtures should include distinct Gello/IK/relay values, delayed and
missing channels, asynchronous ramp logging, short interventions, and an
authority switch inside a nominal 30-step chunk.

Evaluate physically only under an approved supervised protocol, at the fixed
recorded-data baseline. Use repeated comparable initial conditions and report
task success, progress, interventions, completion time, contact/grasp failures,
and tracking/smoothness metrics. A smoother pair of rollouts is useful debugging
evidence, not proof of better training or task performance. Keep all raw data,
manifests, review decisions, and checkpoints so each choice is reversible.

## 8. Running the implemented pipeline

Run commands from this OpenPI checkout in a dedicated **training** environment,
not by upgrading packages in a running Market42 serving environment. The
OpenPI lockfile provides the model/training dependencies; the small
[`requirements-dagger.txt`](../scripts/requirements-dagger.txt) adds MCAP support.
The local serving environment inspected during implementation had protobuf 7
with a logger expecting protobuf 3–6; the locked training environment uses
protobuf 6. Do not copy that serving environment's unpinned overrides.

### Review and convert

Copy [review.example.json](dagger/review.example.json) and populate raw paths,
collection groups, train/validation assignments, and reviewed intervals.
Interval times are seconds **relative to `session_meta.dagger.started_at`**;
they are half-open `[start_s, end_s)`. Keep every episode/clip from one collection
group in one split. Use meaningful session/scene groups, not a different invented
group for every clip. Approved intervals need a reviewer and a reason and must
agree with the recorded authority (`policy` or `teleop`). Unlisted or unapproved
time is excluded. Failed episodes may contain useful approved corrections, but
their policy portions are not automatically good targets.

```bash
uv run --frozen --with-requirements scripts/requirements-dagger.txt \
  python scripts/yam_data/convert_market42_dagger.py \
  --review-manifest /data/review.json --output /data/market42_dagger_round1
```

The output must be new and outside raw episodes. Conversion never overwrites
raw data and publishes `manifest.json` only after all episodes finish. A failed
conversion may leave a partial output directory without a loadable manifest;
use a new output path after correcting the cause. Each export contains:

- A continuous 30 Hz timeline with measured follower state, follower-command
  actions, independent validity/review/authority/segment arrays.
- A synchronized vertical video: top, left wrist, right wrist, each 224 square.
  Both crop and pad match the old converter's PIL BILINEAR operation. v12 training
  requires **all three center-cropped**; its continuation plan rejects other modes.
- Original session metadata, effective node metadata, outcome annotation,
  source hashes, converter hash, review decisions, and explicit thresholds.
- Counts for rejection reasons and eligible 30-action chunks by source, with
  checksums verified by the loader. Counts may overlap across rejection reasons.

The initial conservative limits are: state/relay age 50 ms, producer age 200 ms,
camera age 100 ms, producer/relay joint difference 0.05 rad and gripper difference
0.05. Each segment excludes its recorded ramp duration plus 100 ms at the start,
and 100 ms at the end. These are recorded configuration choices, not universal
safety guarantees; inspect real handoff traces before deliberately changing them
through the review manifest's `options`. Subscriber-driven async command logs,
unknown ramp duration, missing producers, incomplete videos, and backwards
timestamps fail closed. The loader never substitutes raw Gello joints.

### Prepare the old-data split and training plan

[old_splits.example.json](dagger/old_splits.example.json) illustrates the required
mapping from **every** old LeRobot episode index to its original ID, collection
group, and split. Attest the actual preprocessing and leader-target lineage;
do not rename a v11 padded-wrist dataset to v12. The plan verifies complete index
coverage, duplicate IDs, and group leakage, including overlaps with new data.
It cannot detect dishonest or mistaken IDs/group assignments.

An old episode seen while training the initialization checkpoint is not a novel
holdout just because you move it into validation now. Preserve known prior
holdouts and report old-validation exposure; use genuinely held-out new groups
for independent DAgger evaluation. A random old episode split alone does not
prove independence from its collection scene/session.

Fill [plan.example.json](dagger/plan.example.json). `initial_checkpoint` is the
specific checkpoint step directory containing `params/` and `assets/`, not the
model's parent directory. This recipe explicitly supports the audited v12
lineage. It retains its 30-action model, prompt, driver-order absolute 14D action
semantics, and **the selected checkpoint's 32D padded normalization**. All
components must use identical stats and normalization mode. Do not recompute
normalization for this continuation recipe.

Weights are ordered old / new teleop / new policy. `[0.8, 0.1, 0.1]` is the
ABC-inspired default; `[1, 0, 0]` and `[0.8, 0.2, 0]` support controlled baseline
experiments. Zero-weight new sources are disabled. Every enabled source must
have eligible **train and validation** chunks; an empty source is an error,
not a reason to silently redistribute its weight. The old source remains positive
and primary so saved serving assets retain the original identity.

### Validate first, then explicitly train

```bash
uv run --frozen --with-requirements scripts/requirements-dagger.txt \
  python scripts/train_dagger.py --plan /data/plan.json --report /data/preflight.json

uv run --frozen --with-requirements scripts/requirements-dagger.txt \
  python scripts/train_dagger.py --plan /data/plan.json --train
```

Preflight does not load model weights. It validates manifests and all exported
checksums, counts complete chunks per source/split, decodes representative
transformed samples, checks shared normalization, and exercises actual training
and validation mixture batches. It also reports source draws/unique chunk counts
over the first up-to-10,000 virtual catalog entries. This is a sampler audit, not
an actual training-order counter or an exhaustive decode of every old video.
The converter does exhaustively decode all new input videos.
The old-data component explicitly uses PyAV video decoding, avoiding dependence
on a CUDA-linked TorchCodec build just to load training frames.

Training starts from the selected checkpoint **weights**, with a fresh optimizer
and a logged learning-rate budget. It does not restart from pi05_base. Checkpoints
carry `assets/dagger_training.json`, the shared stats, and `prompt.txt`; normal
OpenPI policy loading/Market42 scanning can use the original v12 config name.
The first implementation uses ordinary flow-matching training, not ABC's
prefix-conditioned training objective or timestep-masked imitation.

For an interrupted run, add `--train --resume` with the unchanged plan and input
manifests. A completed checkpoint and matching saved provenance are required.
This restores optimizer/model state; the existing trainer restarts its data
iterator, so it is **not bit-for-bit data-order continuation**. For a new data
round or changed weights/budget, choose a new `exp_name` and initialize from the
previous round's selected checkpoint. Existing experiment directories are never
automatically overwritten.

Validation now evaluates each mixture source separately on its own fixed,
deterministic held-out batches: `val_loss/<source>` per component, with the
aggregate `val_loss` computed as the exact mixture-weighted sum of per-source
means, so a weak minority source (e.g. 10%-weight interventions) is visible
rather than absorbed. These are still imitation losses, not per-controller
success metrics; keep source/profile-stratified physical and offline evaluation
as a release gate. This implementation does not launch a robot trial or upload
datasets.

The converter also records each episode's YAM `gravity_comp_profile` into the
export manifest (`controller_profiles`; `null`/"unrecorded" when the recording
predates the field). The training plan carries the summary into
`assets/dagger_training.json` and into the checkpoint's `policy_metadata`, and
warns when profiles are mixed or not `v8dj_recorded`. Market42-side enforcement
(comparing the serving station's active profile against the checkpoint's
`policy_metadata` before running) is still to be implemented in Lab42.

### Branch integration and test scope

Selectively reused the weighted sampler, loader refactor, normalization-stat
mixture support, and tests from `origin/karim/mixture-datasets` at
`ced6d2c3375cba76c8e055ea85a3b3cb1a747873` (plus its factory definition from the
preceding commit). The subsequent merge of the published
`karim/industrial-packing-v3` tip at `69a014dc4420dae54cec269a3ab73a026b0ae19f`
also preserves its v13 configurations/manifests. Those do not change the
explicit v12 lineage supported by this initial DAgger continuation plan.
Added DAgger-specific validation rather than treating a generic weighted mixture
as an authority-aware dataset. Source probabilities hold in expectation,
independently of dataset sizes, not exactly within every batch.

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 \
  uv run --frozen --with-requirements scripts/requirements-dagger.txt \
  python -m pytest -q scripts/market42_dagger_test.py
```

The smoke test performs real MCAP/video conversion, creates a small old LeRobot
dataset, exercises all enabled sources/splits, trains a tiny pi05 model on CPU,
saves and reloads it for inference, and warm-starts a second training run. Only
the language/vision sizes and tokenizer are substituted; this is wiring coverage,
not a production-v12 training benchmark. Full training-data review, exhaustive
old-dataset integrity checking, and supervised policy evaluation remain required.

Local offline verification on 2026-09-11 also converted these completed raw
recordings without approving any intervals:

| Episode | Timeline frames | Valid frames | Approved chunks |
| --- | ---: | ---: | ---: |
| `episode_20260911_061302_997_a4b5d31b.npy.mp4` (includes intervention) | 5,627 | 5,449 | 0 |
| `episode_20260911_061640_748_020379cd.npy.mp4` (policy only) | 2,851 | 2,802 | 0 |

This verifies actual recorder compatibility and fail-closed review gating,
not demonstration quality. The full v12 LeRobot dataset was not staged at the
usual local dataset root during this check, so production-mixture preflight is
still required on the training machine. The 41 targeted regression tests passed;
the unrelated ALOHA integration test in the broader suite encountered this
machine's TorchCodec/CUDA dependency mismatch. The DAgger path uses direct PyAV
decoding and does not require that backend.
