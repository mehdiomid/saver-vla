# SAVER-VLA project page

## The core idea

Think of SAVER-VLA as a supervisor that sits next to any robot policy. The policy (SmolVLA today) does the normal work. The supervisor watches, and when something goes wrong it pauses the policy, asks the agent what to do, runs a few safe corrective moves, and then hands control back.

None of that logic depends on LIBERO. Today it’s tangled with LIBERO only because the notebook edits LeRobot’s simulator code directly. The package’s job is to separate “the recovery method” from “the robot it runs on,” so the same method works in simulation today and on a real arm later.

## How the package is organized

The package has two halves. The core is the recovery method itself, written once and never tied to a specific robot. The adapters connect it to a particular robot, simulator or model. Moving to a physical robot then means writing new adapters, not rewriting the method.

## The core (stays the same everywhere)

* Verifier: watches the robot’s state each step and answers questions like “is the object still in the gripper?” and “did the last grasp work?”

* Skills: small, bounded corrective moves such as “go above the object,” “approach it,” “shift the grasp a little.” Each has a time limit and reports whether it succeeded.

* Agent: decides which skill to use next. It can be an LLM (GPT-5.5), a fixed script, or a recorded log played back for testing.

* Coordinator: the referee. It tells the agent which skills are allowed right now, checks each request, enforces attempt limits, and logs what was asked for versus what was done.
* Supervisor loop: ties everything together. It runs the policy normally, calls the agent only after a failure, executes skills, and decides when to hand control back.

## The adapters (swapped per setup)

* Robot: sends motion commands and reads the arm’s position and gripper. A LIBERO adapter today; a real-arm adapter later, for example one of the arms LeRobot already supports.

* State estimator: tells the supervisor where the object is. In simulation it reads the exact position from MuJoCo. On a real robot it has to come from cameras, and this is where the YOLO work from Phase 2.4 becomes useful again.

* Policy: wraps the VLA. SmolVLA first, but any VLA that maps an image and instruction to actions can plug in.
Safety layer: limits how far and how fast skills can move, keeps the arm inside a workspace box, and supports an emergency stop and an optional “ask a human before moving” mode. In simulation this is mostly a formality; on hardware it’s required.

## How testing fits in

Testing reliability needs two things the robot itself doesn’t: the ability to cause failures on purpose and a way to measure results over many runs.

* Fault injection: in simulation we can cause the same failure exactly, as many times as we like (drop the can, open the gripper, bias the approach). On a real robot, failures come from a person knocking the object or from natural mistakes, so the package records them rather than creating them.

* Experiment runner: runs many episodes across seeds, fault types and agents, then reports success rates with confidence intervals, how fast failures were detected, how many decisions the agent needed, and how often its requests were rejected.

* Quick tests: check the core pieces against simple fake robots in seconds, with no GPU or simulator. This is what keeps the package trustworthy as it grows.

## What changes on a real robot

The core method transfers as is. These parts need new work:

* Seeing the object: there’s no exact simulator state, so object position comes from cameras with some error. The verifier and skills must tolerate that noise. This is the biggest open research step.

* Calibration: skill distances such as “5 cm above the object” depend on each robot’s movement scale. The Phase calibration becomes a small reusable routine.

* Safety and timing: commands must respect real limits. The agent is called only a few times per episode, so LLM response time isn’t a problem, but the robot must hold still safely while it waits.