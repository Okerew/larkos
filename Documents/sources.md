# Sources

Not a bibliography. Just a note to self of where some of the structure came from so I
stop overexplaining why certain systems look the way they do. Most of this I probably
would have converged on eventually anyway because once you start building persistent
adaptive systems the same architectural patterns keep showing up. If similar ideas
exist elsewhere its because they are natural consequences of the problem space not
because I copied a paper line for line. 
Also a lot of people independently arrive at similar conclusions about hierarchical memory, attractor systems, emotional state spaces, and distributed neuron topology.
That is not surprising. The constraints themselves push you there.

---

### PAD (Pleasure-Arousal-Dominance) — Mehrabian & Russell, 1974

The `EmotionVector` struct uses valence, arousal, and dominance directly as the primary
emotional coordinate system which is basically the PAD model whether intentionally or
not. Three dimensions are enough to produce surprisingly rich affective behavior once
momentum and attractor transitions are introduced. Adding more dimensions mostly just
creates noise and instability unless they are orthogonal.

* `neural_web.c:EmotionVector:2-4` — core emotional axes
* `neural_web.c:updateEmotionMomentum():1-17` — emotional inertia and smoothing
* `neural_web.c:updateAttractorDynamics():1-70` — transitions between emotional basins

### Emotion Attractor Dynamics — Kuppens, Oravecz, & Tuerlinckx, 2010; Lewis, 2005

The emotional system behaves more like a dynamical landscape than a clean state machine.
States pull toward stable basins and certain emotional trajectories become easier to
re-enter over time. This is where the pathological reinforcement logic came from.
Depression anxiety obsession etc all look computationally like attractors with excessive
reinforcement and reduced escape probability.

The important realization was that emotions are not events they are gravitational fields.

* `neural_web.c:EmotionAttractor:1-17` — basin strengths and transition topology
* `neural_web.c:findNearestAttractor():1-30` — distance and basin pull calculations
* `neural_web.c:updateAttractorDynamics():1-70` — entry exit and reinforcement logic

### Attachment Theory — Bowlby, 1969; Ainsworth, 1978

`AttachmentBond` is basically attachment theory translated into persistent weighted state.
Repeated interaction increases familiarity but trust updates asymmetrically because trust
is easier to destroy than build. Conflict leaves residue. Separation has computational
cost. Long term interaction history matters more than single events.

This ended up producing much more believable social persistence than simple affinity
scores ever did.

* `neural_web.c:AttachmentBond:1-17` — bond representation
* `neural_web.c:updateAttachmentBond():1-12` — trust familiarity and attachment updates
* `neural_web.c:findOrCreateBond():1-31` — initialization and persistence logic

### Self-Verification Theory — Swann, 1983

The identity system keeps a reference state and periodically checks current behavior
against it. If drift exceeds threshold the system flags instability and can revert or
self-correct. That maps almost directly onto self verification theory where agents try
to preserve continuity of self perception over time.

Without this kind of mechanism long running adaptive systems tend to dissolve into
behavioral noise after enough updates.

NOTE: Although I came to these conclusions myself, I used this work as a reference for how to implement them.

* `neural_web.c:SelfIdentitySystem:22-27` — verification subsystem
* `neural_web.c:verifyIdentity():1-29` — consistency evaluation
* `neural_web.c:generateIdentityReflection():19-20` — reflection and coherence scoring

### Self-Complexity Theory — Linville, 1985

The `self_complexity` field exists because psychologically simple systems are fragile.
If identity depends on too few emotional or conceptual regions then perturbations become
catastrophic. Systems with multiple semi independent self aspects handle stress better
because damage localizes instead of globally propagating.

Higher complexity here is not randomness its compartmentalized resilience.

NOTE: Although I came to these conclusions myself, I used this work as a reference for how to implement them.

* `neural_web.c:AffectiveSystem:13` — complexity metric
* `neural_web.c:updateAffectiveComplexity():1-40` — state diversity and conflict analysis

### Theory of Mind — Premack & Woodruff, 1978; Baron-Cohen, 1995

The social system maintains internal models of other agents instead of treating them as
static entities. Prediction confidence relationship quality inferred traits and emotional
alignment all update continuously. Incorrect predictions force model revisions.

Once this layer exists social interaction stops looking reactive and starts looking
anticipatory.

* `neural_web.c:PersonModel:1-9` — inferred traits and prediction state
* `neural_web.c:SocialSystem:1-22` — empathy and behavioral modeling
* `neural_web.c:updateEmpathy():1-7` — emotional difference adjustment logic

### Functional Specialization — Hubel & Wiesel, 1962; Fodor, 1983

The specialization system emerged from observing that generalized neurons waste compute.
Over time neurons naturally drift toward roles they are statistically useful at. Pattern
detectors become better pattern detectors temporal processors become better temporal
processors and so on.

This is less about hard modularity and more about adaptive efficiency gradients.

* `neural_web.c:NeuronSpecializationType:1-11` — specialization categories
* `neural_web.c:initializeSpecializationSystem():1-18` — specialization bootstrap logic
* `neural_web.c:detectSpecializations():1-50` — specialization scoring from activation history

