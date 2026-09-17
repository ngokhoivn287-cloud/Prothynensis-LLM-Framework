# MMNs 7.x — Massive Model Networks Architecture

**Product:** Prothynesis 3.6  
**Version:** 7.x  
**Status:** Design Document  
**Date:** 2026-09-16

---

## Philosophy

The central hypothesis of MMNs 7.x is:

> **Collective capability emerges from a network of highly capable specialists, not from a bag of shallow generalists.**

Previous architectures treated every Solver as a broad generalist. This approach wastes capacity on domains that other Solvers already cover better.

The new philosophy:

- **BREADTH = population**
- **DEPTH = individual Solver**
- **COLLECTIVE INTELLIGENCE = MMNs**

Each Solver is allowed to learn only a subset/slice of the total training distribution. But that slice must be learned extremely deeply, accurately, and robustly.

A 150M Solver that deeply masters its assigned domain is preferred over a 150M Solver that is mediocre at many unrelated domains.

The population provides breadth. Each individual Solver provides depth.

---

## 150M Solver Architecture

### Parameter Budget

| Component | Parameter Count | Percentage |
|-----------|----------------|------------|
| Core General Intelligence Layer | ~30M | 20% |
| Specialized Mastery Layer | ~105M | 70% |
| Transfer / Generalization Layer | ~15M | 10% |
| **Total** | **~150M** | **100%** |

### Layer 1 — Core General Intelligence

Shared across the whole population. Every Solver must learn enough of:

- instruction following
- language understanding
- reasoning
- abstraction
- decomposition
- planning
- basic mathematics
- basic coding
- debugging
- verification
- uncertainty estimation
- tool semantics
- computer-use semantics
- CRW protocol
- collaboration protocol

This core should be compact and high-quality.

**Implementation:**  
The core is a fixed, pretrained subnetwork that is shared across all Solvers. During specialization training, the core may be optionally frozen or trained with a very low learning rate to preserve general capabilities.

### Layer 2 — Specialized Mastery

Each Solver receives a selected high-value subset of the global training distribution.

Examples of specialization domains:

- systems programming
- algorithms
- mathematics
- security
- OS design
- compilers
- databases
- distributed systems
- scientific reasoning
- planning
- software engineering
- formal logic
- chemistry
- physics
- biology
- finance
- legal reasoning
- debugging
- testing
- adversarial reasoning
- verification
- research synthesis

The specialization must be based on **measurable capability coverage**, not arbitrary labels.

### Layer 3 — Transfer / Generalization

The remaining fraction allows the Solver to generalize beyond its narrow specialization. This prevents catastrophic forgetting and enables:

- cross-domain reasoning
- adaptation to unfamiliar surface forms
- basic tool use
- computer interaction
- collaboration with other Solvers

Exact ratios MUST be validated experimentally.

---

## Data Allocation

### Global Data Pools

The training system maintains the following global data pools:

1. **GLOBAL CORE DATA** — Common to all Solvers
   - Instruction following
   - Language understanding
   - Basic reasoning
   - Mathematics fundamentals
   - Coding fundamentals
   - Tool-use basics
   - Computer-use basics
   - Collaboration protocol
   - CRW protocol

2. **SPECIALIZED DATA** — Domain-specific
   - Curated for each specialization
   - High difficulty
   - High information density
   - Verifiable solutions

3. **TRANSFER DATA** — Cross-domain generalization
   - Examples that bridge domains
   - Novel problem types
   - Edge cases

4. **VERIFICATION DATA** — Self-verification training
   - Correct/incorrect pairs
   - Error detection
   - Proofreading
   - Testing

5. **ADVERSARIAL DATA** — Robustness training
   - Paraphrases
   - Distribution shifts
   - Adversarial examples
   - Edge cases

6. **TOOL-USE DATA** — Tool interaction training
   - Tool selection
   - Tool composition
   - Tool output interpretation
   - Failure handling

7. **COMPUTER-USE DATA** — Computer interaction training
   - Filesystem operations
   - Shell commands
   - Application interaction
   - Screen inspection
   - State verification

8. **PLANNING DATA** — Planning and execution
   - Task decomposition
   - Dependency identification
   - Execution planning
   - Intermediate verification
   - Recovery from failure

9. **COLLABORATION DATA** — Multi-Solver interaction
   - CRW message formats
   - Handoff protocols
   - Context sharing
   - Conflict resolution

### Per-Solver Manifest

Every Solver receives a manifest describing:

- core data allocation
- specialized domains
- specialization depth
- transfer examples
- difficulty distribution
- reasoning patterns
- verification patterns
- tool-use tasks
- computer-use tasks

Example manifest:

```json
{
  "solver_id": "0001842",
  "core_data": {
    "allocation_percentage": 20,
    "datasets": ["global_core_v1"],
    "token_budget": 20000000
  },
  "specialization": {
    "primary_domain": "software_engineering",
    "allocation_percentage": 70,
    "datasets": ["swe_specialized_v2", "code_review_v1"],
    "token_budget": 70000000,
    "depth_target": "expert"
  },
  "transfer": {
    "allocation_percentage": 10,
    "datasets": ["transfer_general_v1"],
    "token_budget": 10000000
  },
  "tool_use": {
    "allocation_percentage": 0,
    "datasets": ["tool_use_core_v1"],
    "token_budget": 0
  },
  "computer_use": {
    "allocation_percentage": 0,
    "datasets": ["computer_use_core_v1"],
    "token_budget": 0
  }
}
```

---

## Training Strategy

### Capability-Based Stopping

Do NOT force every Solver to stop at exactly the same step count.  
Do NOT assume same steps = same quality.  
Do NOT assume same tokens = same quality.

Use capability-based stopping.

**Stopping conditions:**

- domain benchmark plateau
- general reasoning plateau
- verification plateau
- diminishing marginal gain
- overfitting detection
- robustness regression

**Record per Solver:**

- training tokens
- steps
- specialization tokens
- core tokens
- validation metrics
- domain mastery score
- general capability score
- robustness score

### Training Stages

1. **Stage 1: Core Intelligence Initialization**
   - Train core layer on global core data
   - Ensure general capability baseline
   - Validate reasoning, planning, tool use

2. **Stage 2: Specialization Acquisition**
   - Introduce specialized data
   - Begin domain-specific training
   - Monitor domain mastery metrics

3. **Stage 3: Specialization Deepening**
   - Increase specialization data ratio
   - Focus on difficult examples
   - Improve robustness

4. **Stage 4: Transfer Learning**
   - Introduce transfer data
   - Cross-domain examples
   - Novel problem types

5. **Stage 5: Tool-Use Training**
   - Tool selection and composition
   - Output interpretation
   - Failure handling

6. **Stage 6: Computer-Use Training**
   - Filesystem operations
   - Shell commands
   - Application interaction
   - State verification

7. **Stage 7: Planning / Long-Horizon Training**
   - Multi-step planning
   - Execution-grounded planning
   - Intermediate verification
   - Recovery from failure

8. **Stage 8: Verification / Adversarial Training**
   - Self-verification
   - Error detection
   - Adversarial robustness
   - Edge case handling

9. **Stage 9: Population Interaction**
   - CRW protocol training
   - Collaboration patterns
   - Context sharing

10. **Stage 10: Collective Refinement**
    - Multi-Solver tasks
    - Synthesis training
    - Final validation

### Specialization Mastery Metrics

Do NOT use raw training loss as proof of mastery.

Define **SPECIALIZATION MASTERY** using measurable criteria:

- high in-domain accuracy
- robustness to paraphrase
- robustness to distribution shift
- multi-step reasoning
- edge-case handling
- adversarial resistance
- verification ability
- ability to explain / justify
- ability to transfer domain knowledge
- ability to detect uncertainty
- ability to solve novel examples

Each metric is tracked per Solver per domain.

### Data Quality Principles

Data should be:

- correct
- diverse
- difficult
- useful
- verifiable
- non-redundant
- capability-targeted

Avoid low-value token inflation. The objective is **MAXIMUM INFORMATION PER TRAINING TOKEN**.

### Targeted Data Generation Loop

After evaluation:

1. Identify exact capability failures
2. Generate new data targeting:
   - weak concepts
   - common error patterns
   - failure modes
   - missing reasoning strategies
   - blind spots
   - tool-use mistakes
   - planning failures
   - verification failures
3. Train on targeted data
4. Re-evaluate
5. Repeat

---

## Tool Use

Every Solver must understand a common tool protocol.

### Minimum Tool Set

- calculator
- code execution
- filesystem
- shell
- search/retrieval
- structured APIs
- browser/computer interaction

### Tool-Use Training Requirements

- choosing the right tool
- composing tools
- interpreting tool output
- handling failures
- validating results
- recovering from errors
- avoiding unnecessary calls

### Tool Protocol

Tools are invoked via structured calls:

```json
{
  "tool": "filesystem.read_file",
  "arguments": {
    "path": "/home/user/project/main.py"
  }
}
```

Responses:

```json
{
  "tool": "filesystem.read_file",
  "result": {
    "content": "...",
    "size": 1024,
    "success": true
  }
}
```

Errors:

```json
{
  "tool": "filesystem.read_file",
  "error": {
    "code": "FILE_NOT_FOUND",
    "message": "File not found: /home/user/project/main.py"
  }
}
```

---

## Computer Use

Every Solver must have strong computer interaction capability.

### Minimum Computer-Use Skills

- opening files
- editing files
- navigating directories
- running commands
- inspecting logs
- running tests
- browser navigation
- interacting with applications
- reading screenshots
- verifying final state

### Computer-Use Training

Prefer state-grounded tasks over textual imitation.

Examples:

- "Open the file `/tmp/test.py`, add a docstring, save it"
- "Run the test suite and report failures"
- "Navigate to the project root, list files, find the main module"
- "Open the browser, navigate to example.com, take a screenshot"

---

## Planning

Every Solver must understand planning and execution.

### Planning Requirements

1. task understanding
2. decomposition
3. dependency identification
4. ordering
5. execution planning
6. intermediate verification
7. recovery from failure
8. completion checking

### Planning Training

Include execution-grounded planning examples:

```
Task: Implement a REST API endpoint for user registration

Plan:
1. Create file `api/routes/auth.py`
2. Import required modules (FastAPI, models, auth)
3. Define Pydantic schema for registration request
4. Implement password hashing utility
5. Implement registration endpoint with validation
6. Add error handling for duplicate users
7. Write unit tests for the endpoint
8. Run tests to verify

Execution:
[Each step is executed and verified before proceeding]
```

Do NOT make planning merely a textual "plan generation" task.

---

## Verification

Every Solver must have strong verification capability.

### Verification Requirements

A specialized Solver should be able to:

- test
- inspect
- compare
- cross-check
- identify contradictions
- search for counterexamples
- estimate uncertainty
- request another Solver
- revise its belief

### Verification Training

Train Solvers to verify their own work:

- "Write a function, then write tests for it"
- "Solve a math problem, then verify the answer"
- "Generate code, then review it for bugs"
- "Make a claim, then find evidence for/against it"

---

## Stateful Memory

Preserve stateful memory across interactions.

### Memory Types

- **Working Memory** — Current task context
- **Task Memory** — Previous task solutions
- **Episodic Memory** — Past interactions and outcomes
- **Peer Memory** — Other Solvers' outputs and expertise
- **Long-Term Memory** — Accumulated experience

### Memory Usage

Memory helps:

- avoid repeated errors
- recall previous tool interactions
- reuse proven strategies
- recognize familiar failure modes
- improve planning

---

## CRW

Specialized Solvers must communicate through CRW (Collective Reasoning Web).

### CRW Message Types

A Solver should be able to publish:

- hypotheses
- evidence
- arguments
- counterarguments
- uncertainty
- verification results
- tool observations
- plans
- discovered facts
- task state
- warnings
- handoff information

### CRW Protocol

Messages are structured:

```json
{
  "message_id": "msg_001",
  "sender": "solver_0001842",
  "recipient": "solver_0001856",
  "type": "hypothesis",
  "content": {
    "claim": "The bug is in the authentication middleware",
    "evidence": ["Stack trace shows...", "Log indicates..."],
    "confidence": 0.85
  },
  "timestamp": "2026-09-16T10:00:00Z"
}
```

A Solver should be able to retrieve relevant context without requiring the entire population state in its local context window.

---

## Dynamic Recruitment

The Scheduler must learn: **WHICH SPECIALIST SHOULD BE USED FOR WHICH TASK?**

### Recruitment Factors

Recruitment should consider:

- specialization match
- historical performance
- task similarity
- current uncertainty
- verification need
- tool capability
- independence from existing Solvers
- information gain

Do NOT simply select Solvers by fixed domain labels.

### Population Composition

The population should contain:

- broad specialists
- deep specialists
- verification specialists
- adversarial specialists
- planning specialists
- tool specialists
- computer-use specialists
- research/synthesis specialists

Every Solver retains the general intelligence substrate.

---

## Hierarchy

Preserve the hierarchical architecture.

### Levels

1. **SOLVER** — deep expert reasoning
2. **ORCHESTRAL** — specialist coordination
3. **CHIEF** — cross-specialty synthesis
4. **MASTER** — high-level strategy / conflict resolution
5. **ULTIMATE** — final collective reasoning / synthesis

### Responsibilities

| Level | Responsibility |
|-------|---------------|
| Solver | Deep expert reasoning in assigned domain |
| Orchestral | Coordinate multiple Solvers in same domain |
| Chief | Synthesize across different specialties |
| Master | High-level strategy, conflict resolution |
| Ultimate | Final collective reasoning / synthesis |

Do NOT let hierarchy become a simple voting tree.

---

## Population Composition

### Specialist Types

The population should behave like **A NETWORK OF HIGHLY CAPABLE SPECIALISTS**, not a bag of shallow experts.

### Diversity Principles

Differences should primarily arise from:

- training distribution
- specialization
- reasoning experience
- failure history
- memory
- tool experience
- task history

Do NOT create artificial personalities merely to make Solvers "different."

---

## Evaluation

### Multi-Tier Evaluation Suite

**TIER 1: Basic Capability**
- Language understanding
- Basic reasoning
- Simple QA

**TIER 2: Advanced Reasoning**
- Multi-step reasoning
- Logical deduction
- Mathematical problem solving

**TIER 3: Expert Domain Tasks**
- Domain-specific benchmarks
- Professional-level problems
- Certification-style tasks

**TIER 4: Tool / Computer Tasks**
- Tool selection and use
- File operations
- Code execution
- Browser interaction

**TIER 5: Long-Horizon Tasks**
- Multi-step planning
- Extended execution
- State management

**TIER 6: Multi-Agent Collective Tasks**
- CRW communication
- Collaborative problem solving
- Synthesis

**TIER 7: Adversarial / Robustness Tests**
- Paraphrase robustness
- Distribution shift
- Adversarial examples

### Comparison Metrics

For each task, compare:

- best individual Solver
- top-k Solvers
- population Oracle
- CRW
- full MMNs
- single-model baseline where appropriate

### Success Criteria

The architecture should only be considered successful if:

**COLLECTIVE GAIN > 0** on meaningful tasks.

Primary objective: **CAPABILITY**  
Secondary: **EFFICIENCY**  
Third: **SCALABILITY**

Do NOT define success as "MMNs has more total parameters."

---

## Community Training

### Runtime Integration

The Prothynesis Runtime Worker supports the new training paradigm:

- receives Solver ID
- specialization profile
- dataset shard
- training config
- token budget
- evaluation config

Worker returns:

- final checkpoint
- training metadata
- validation results
- benchmark results
- checksum

### Independent Solver Training

Every Solver is independently trainable:

- no synchronous all-population training
- no weight averaging between Solvers
- asynchronous training
- checkpoint registry
- resumable jobs

### Coordinator Verification

Before adding a Solver to the population, the coordinator verifies:

- checkpoint integrity
- Solver identity
- job identity
- runtime compatibility
- metadata consistency
- training token count
- validation result
- artifact integrity

---

## Runtime Integration

### Local Inference

The Runtime must support:

- loading individual Solvers
- running single-Solver inference
- running MMNs inference (multi-Solver)
- CRW communication
- tool use
- computer use

### Web UI Updates

Add to the Web UI:

- Solver specialization display
- Population browser
- CRW message viewer
- Tool-use interface
- Computer-use interface
- Planning interface

### API Updates

Add endpoints:

- `GET /v1/solvers` — list available Solvers
- `POST /v1/solvers/{id}/chat` — chat with specific Solver
- `POST /v1/mmns/chat` — MMNs multi-Solver chat
- `GET /v1/mmns/status` — MMNs status
- `POST /v1/tools/execute` — execute tool
- `POST /v1/computer/execute` — execute computer action

---

## Pilot Experiment

### Design

Create a pilot with **8 Solvers × 150M parameters**.

### Solver Profiles

| Solver | Primary Specialization | Secondary Focus |
|--------|------------------------|-----------------|
| A | Systems / Low-level | OS design, memory management |
| B | Math / Formal Reasoning | Proofs, logic, abstract algebra |
| C | Software Engineering | Design patterns, refactoring, testing |
| D | Security / Adversarial | Penetration testing, cryptography |
| E | Planning / Agents | Multi-step planning, tool use |
| F | Science / Research | Physics, chemistry, research synthesis |
| G | Verification | Testing, debugging, proof verification |
| H | Broad Technical Generalist | Cross-domain, tool use, computer use |

### Measurement Plan

Measure:

- specialist mastery per domain
- general capability retention
- tool use proficiency
- planning ability
- verification ability
- cross-domain transfer
- answer diversity
- error diversity
- pairwise error correlation
- population Oracle improvement
- collective gain

### Protocol

1. Train all 8 Solvers on their specialized manifests
2. Evaluate each Solver individually on:
   - specialization domain benchmarks
   - general capability benchmarks
   - tool-use benchmarks
   - planning benchmarks
   - verification benchmarks
3. Evaluate population-level:
   - top-k Oracle
   - CRW-based collaboration
   - full MMNs synthesis
4. Compare against single-model baseline
5. Analyze failure modes
6. Generate targeted data
7. Retrain and re-evaluate

---

## Scaling Plan

### Phase 1: Pilot (8 Solvers)
- Validate architecture
- Measure collective gain
- Identify failure modes
- Optimize data allocation

### Phase 2: Small Population (64 Solvers)
- Expand specialization coverage
- Test hierarchy layers
- Validate community training

### Phase 3: Medium Population (1024 Solvers)
- Full hierarchy deployment
- Distributed training
- Production coordinator

### Phase 4: Large Population (10,240 Solvers)
- Full Mini population
- Maximum breadth
- Global deployment

Do NOT skip phases. Validate at each stage.

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| No collective gain | HIGH | Early evaluation, fallback to broader training |
| Specialization collapse | HIGH | Capability-based stopping, transfer data |
| Core capability degradation | MEDIUM | Freeze core during specialization, core validation |
| Training instability | MEDIUM | Careful learning rate scheduling, capability monitoring |
| Coordination overhead | MEDIUM | Efficient CRW, context pruning |
| Data contamination | LOW | Strict data versioning, per-Solver manifests |
| Evaluation gaming | MEDIUM | Diverse evaluation, held-out benchmarks |

---

## Open Research Questions

1. What is the optimal core/specialization/transfer ratio?
2. How much transfer data is needed to prevent catastrophic forgetting?
3. What is the minimum core size for general capability?
4. How do we measure "specialization mastery" reliably?
5. What is the best stopping criterion for capability-based training?
6. How much diversity is needed in the population for collective gain?
7. What is the optimal hierarchy depth for 150M Solvers?
8. How do we handle overlapping specializations?
9. What is the cost of CRW communication at scale?
10. How do we prevent solver collapse into similar solutions?

---

## Naming Consistency

- **Product:** Prothynesis 3.6
- **Architecture:** MMNs 7.x (Massive Model Networks)
- **Unit:** Solver
- **Population:** MMNs
- **Protocol:** CRW (Collective Reasoning Web)
- **Hierarchy:** Solver → Orchestral → Chief → Master → Ultimate

Historical MoMMs 3.4 artifacts remain historical and should not be rewritten in a way that breaks reproducibility.

---

## Implementation Status

- [x] Architecture design document
- [ ] Core Solver architecture (150M)
- [ ] Data allocation system
- [ ] Capability mastery metrics
- [ ] Tool/computer-use interfaces
- [ ] Specialization scheduler
- [ ] Per-Solver evaluation
- [ ] Population-level evaluation
- [ ] 8-Solver pilot training
- [ ] Pilot evaluation and analysis
