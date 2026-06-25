## Introduction: The Epistemological Crisis in Artificial Cognition

The pursuit of artificial general intelligence has historically oscillated between two dominant paradigms, neither of which fully captures the fluid, embodied, and highly efficient nature of biological cognition. On one side, the symbolic approach, epitomized by Jerry Fodor’s Language of Thought Hypothesis (LOTH) or "Mentalese," posits that cognition operates via language-like representations governed by rigid syntactic and logical rules. While LOTH excels at explaining the compositionality and productivity of thought, it fails to ground symbols in sensory reality and struggles to model the continuous, analog phenomena of the physical world. On the other side, the connectionist approach models intelligence through the continuous, high-dimensional weight updates of artificial neural networks. While connectionism has achieved remarkable success in pattern recognition and generative tasks, it suffers from profound architectural limitations. Modern deep learning networks require exorbitant computational resources, struggle with abstract reasoning, and are notoriously susceptible to "catastrophic forgetting"—the tendency to rapidly overwrite previously acquired knowledge when exposed to new data distributions. Furthermore, the representations within these models lack a grounding in the physical world, operating as unmoored statistical associations rather than true, actionable cognitive units.   

To construct a computational architecture capable of lifelong, continuous learning with extreme energy efficiency, a fundamental reconceptualization of the foundational cognitive unit is required. This analysis proposes the formalization of the pre-linguistic cognitive unit 'X'—a foundational element of meaning that precedes formalized language, grounded in sensorimotor experience rather than arbitrary syntax. Rather than modeling 'X' through the continuous trajectories of gradient descent or the rigid syntax of Mentalese, this framework grounds the architecture of 'X' in the Buddhist Abhidhamma. The Abhidhamma provides a highly sophisticated, phenomenological psychology that views consciousness not as a continuous, unbroken stream, but as a discrete, event-driven sequence of momentary mental states (_cittavithi_).   

By translating this ancient model of discrete, moment-by-moment perception into modern mathematical structures—specifically utilizing the geometry of Conceptual Spaces, the algebra of Vector Symbolic Architectures (VSAs), the dynamical governance of Conceptors, and the optimization mechanics of the Free Energy Principle—this report outlines a comprehensive neuro-symbolic framework for 'X'. This architecture mimics the biological mechanisms of the human brain, utilizing Spiking Neural Networks (SNNs) and localized, gradient-free synaptic plasticity to achieve continuous intellectual improvement with an ultra-low computational footprint.   

## 1. The Ontological Foundation: Discretizing the Stream of Consciousness

### 1.1 The Paramattha Dhammas and the Architecture of Citta

In the Theravada Abhidhamma, the familiar world of substantial objects and enduring persons is viewed as a conceptual construct fashioned by the mind. Ultimate reality, by contrast, is deconstructed into fundamental constituents known as _paramattha dhammas_. Unlike conventional reality, which relies on linguistic labels, ultimate reality is comprised of elementary phenomena that flash into and out of existence in a fraction of a millisecond. The conditioned realities are divided into three primary categories: _Citta_ (consciousness or bare awareness), _Cetasika_ (mental factors that accompany consciousness), and _Rupa_ (material or physical phenomena).   

_Citta_ is the core faculty of cognition, defined phenomenologically as the event of knowing or experiencing an object. However, _citta_ never arises in isolation. It is perpetually accompanied by _cetasikas_—mental factors that color, direct, and define the specific computational nature of that cognitive moment. While a _citta_ provides the foundational substrate of awareness, the _cetasikas_ execute specific cognitive functions, providing feeling (_vedana_), perception (_sanna_), volition (_cetana_), and various other wholesome, unwholesome, or neutral qualities. For example, a cognitive unit rooted in craving (_lobha mula citta_) accompanied by a pleasant feeling (_somanassa_) and unprompted volition (_asankharika_) represents a highly specific, multidimensional mental state.   

Crucially, the Abhidhamma posits a strict doctrine of momentariness (_khanika-vada_). Consciousness is not a sustained, fluid continuum; it is a highly discrete, automaton-like sequence of events. Each _citta_ arises, performs its specific function, and perishes instantaneously, immediately transmitting its causal momentum to the subsequent _citta_. This ontological stance aligns remarkably well with Spontaneous Collapse Theory in quantum mechanics, which posits a "flash ontology" where physical reality is a sparse galaxy of discrete, point-like events rather than a solid substance. For artificial intelligence, this provides a profound architectural blueprint: a cognitive engine need not maintain a massive, constantly active dense network computing continuously. Instead, cognition can be modeled as a sparse, event-driven sequence of discrete computational states, vastly reducing energy expenditure.   

### 1.2 The Cittavithi: The 17-Moment Cognitive Sequence

When an external stimulus impinges upon the senses, the mind does not process it uniformly. According to the Abhidhamma, a complete sensory cognitive process (_pancadvara-cittavithi_) unfolds across a strictly defined sequence of seventeen distinct thought-moments (_cittakkhanas_). Prior to the introduction of a stimulus, the mind rests in a baseline, passive state known as _bhavanga_ (the life-continuum). The _bhavanga_ functions similarly to an idling CPU; it maintains the continuity of existence without actively processing new external data.   

The 17-moment cycle executes as a highly structured, finite state machine: The cycle begins with the interruption of the baseline. When a stimulus strikes, the _bhavanga_ vibrates for one moment (_bhavanga calana_) and is subsequently arrested (_bhavanga upaccheda_). Following this, the mind turns toward the stimulus at the specific sense door (_pancadvaravajjana_), receiving the raw sensory data, and then actively investigating it (_santirana_).   

The critical pivot of the sequence occurs at the eighth moment, the Determining phase (_Votthapana_). Here, the mind classifies the object and establishes the necessary psychological posture to respond to it. This moment acts as a strict gating mechanism; it dictates whether the subsequent active processing phases will execute or whether the stimulus is deemed inconsequential, allowing the system to return to the _bhavanga_ state.   

If the stimulus is salient, the sequence enters the Impulsion (_Javana_) phase. This is the dynamic, active phase of cognition, consisting of seven consecutive, identical _cittas_ that "run" over the object. It is during the _javana_ phase that semantic meaning is actively processed, ethical or pragmatic valuation is applied, and volitional action (_kamma_) is generated. Following the impulsion, the Registration (_Tadarammana_) phase occurs for two moments, allowing the experience to be consolidated into the cognitive stream before the mind sinks back into the _bhavanga_ state.   

This 17-moment model provides a rigorous control-flow mechanism for the pre-linguistic cognitive unit 'X'. It suggests a neuromorphic architecture where continuous environmental monitoring is handled by ultra-low-power baseline states, and dense, active computation is deployed recursively only during the highly localized _javana_ phase.   

|Cognitive Paradigm|Foundational Unit|Processing Mechanism|Hardware Analogue|
|---|---|---|---|
|**Classical Symbolic (LOTH)**|Linguistic Tokens (Mentalese)|Syntactic rule application, formal logic|CPU / Turing Machine|
|**Standard Connectionism**|Continuous Real-Valued Vectors|Global gradient descent, dense matrix multiplication|GPU / Dense Accelerators|
|**Abhidhamma-Neurosymbolic**|Discrete 'X' (Image Schema bound to Cetasikas)|Sparse, 17-moment event-driven sequence (_Cittavithi_)|Neuromorphic Spiking Neural Networks|

  

## 2. The Geometry of Meaning: Conceptual Spaces and Latent Convexity

To instantiate the discrete phenomenology of the Abhidhamma into a functional computational model, the representation of the pre-linguistic unit 'X' must be formalized. Traditional word embeddings map syntactic tokens to vectors, which successfully capture distributional semantics but fail to capture the embodied, pre-linguistic nature of raw cognition. Instead, 'X' must be modeled using the mathematics of Conceptual Spaces.   

### 2.1 Perceptual Symbol Systems and Image Schemas

Before formalized language exists, cognition is fundamentally grounded in sensorimotor experience. The theory of Perceptual Symbol Systems (PST), primarily developed by Lawrence Barsalou, posits that all cognitive processes are grounded in sensory and motor systems. Instead of storing abstract, amodal symbols in a disconnected semantic memory, the brain reactivates and recombines the perceptual and action states experienced when encountering a concept. These mental simulations form the basis of meaning, suggesting that cognition is deeply intertwined with how an agent physically interacts with its environment.   

These simulations aggregate into foundational cognitive building blocks known as "Image Schemas." An image schema is a condensed, dynamic redescription of perceptual experience that maps spatial and temporal structures onto conceptual thought. Learned in early infancy through bodily interactions, image schemas exist as spatiotemporal relationships that enable action and describe characteristics of the environment. Common examples include CONTAINMENT (the notion of an inside, a boundary, and an outside), SOURCE-PATH-GOAL (movement from a starting point to an end point), and VERTICALITY (spatial orientation tied to gravity). These pre-linguistic structures provide the scaffolding for later abstract reasoning and conceptual metaphor. For instance, the physical experience of CONTAINMENT maps to the abstract linguistic concept of being "in" a state of depression, while VERTICALITY maps to power dynamics (e.g., "ranking high"). In the framework of 'X', an Image Schema serves as the fundamental _Rupa_ (material/sensory object) being processed by the _Citta_.   

### 2.2 The Mathematics of Conceptual Spaces

To map these pre-linguistic Image Schemas mathematically, the architecture utilizes Peter Gärdenfors' theory of Conceptual Spaces, which serves as a robust bridge between symbolic and sub-symbolic representations. A conceptual space is a geometric structure defined by a set of quality dimensions that represent interpretable features of the environment. These dimensions can range from low-level perceptual inputs (e.g., hue, saturation, brightness, temperature, spatial coordinates) to higher-level abstract qualities.   

In this geometry, instances or specific observations are represented as points within the space, while broader concepts are represented as regions. A central tenet of Gärdenfors' theory, known as Criterion P, asserts that natural categories and concepts correspond strictly to convex regions within a conceptual space. The property of convexity dictates that if two points, x and y, belong to a specific conceptual category representing 'X', any point z that lies between them on a linear segment must also belong to that category. This formulation naturally supports prototype theory; the focal point or center of gravity of the convex region serves as the category prototype, and semantic similarity is calculated merely as the inverse of the geometric distance from this prototype.   

Recent analyses of the latent spaces within modern deep neural networks have empirically validated the importance of this geometry. When neural networks learn robust representations through self-supervised pretraining, the resulting latent spaces naturally organize into convex conceptual regions. Furthermore, the degree of graph convexity within these latent regions strongly correlates with the model's generalizability, its capacity for few-shot learning, and its alignment with human conceptual representations. By enforcing geometric convexity in the latent representation of 'X', the proposed architecture inherently supports categorical inference and similarity matching, bypassing the need for rigid symbolic rules while maintaining high interpretability.   

## 3. The Algebra of the Mind: Vector Symbolic Architectures for Citta and Cetasika

While Conceptual Spaces define the continuous geometry of the sensory object (the Image Schema), a mechanism is required to bind this object to the active mental state and compute the discrete transitions of the 17-moment sequence. Vector Symbolic Architectures (VSAs), also known as Hyperdimensional Computing (HDC), provide the ideal computational algebra to manipulate these representations without succumbing to the combinatorial explosion typical of symbolic AI.   

In VSAs, information is represented using extremely high-dimensional vectors (e.g., D=10,000), where semantic meaning is distributed holographically across the entire array of numbers. Because the dimensionality is so vast, any two randomly generated vectors are nearly orthogonal to one another, providing an immense capacity for distinct, interference-free representations.   

The composition of a cognitive moment in the Abhidhamma—where a single _citta_ is simultaneously accompanied by multiple varying _cetasikas_ (mental factors)—maps seamlessly to VSA operations. VSAs utilize three primary operations that allow for complex compositionality while strictly preserving the fixed dimensionality of the vector space:   

|VSA Operation|Mathematical Function|Abhidhamma Application|Cognitive Outcome|
|---|---|---|---|
|**Binding (⊗)**|Element-wise multiplication|Linking _Citta_ to _Cetasika_<br><br>[cite: 22]|Creates a novel, composite state (e.g., Awareness + Pleasant Feeling) that is nearly orthogonal to its constituent parts, representing a unique experiential moment.|
|**Bundling (⊕)**|Element-wise addition|Aggregating multiple _Cetasikas_<br><br>[cite: 23]|Superimposes all active mental factors into a single memory trace for the cognitive moment. The resulting vector remains highly similar to its components, allowing for retrieval of individual factors.|
|**Permutation (Π)**|Coordinate shifting / rotation|Ordering the _Cittavithi_ sequence|Preserves the strict temporal order of the 17-moment sequence. Ensures that "Determining follows Investigating" remains distinct from a reversed sequence.|

  

Let C be the base consciousness vector (_citta_), S be the geometrically derived sensory object (the Image Schema), and {F1​,F2​,...,Fk​} be the set of active mental factors (_cetasikas_). The mathematical representation of the complete pre-linguistic cognitive unit 'X' at a discrete time step t within the _cittavithi_ is formalized as:

Xt​=(C⊗S)⊕i=1∑k​(C⊗Fi​)

This high-dimensional formulation ensures that 'X' is extraordinarily robust to noise. Unlike traditional artificial neural networks, where minor perturbations can cause catastrophic failure, a VSA vector can tolerate significant bit-flip errors or signal corruption. The holographic nature of the representation means that even a degraded vector can still be accurately decoded to its correct constituent concepts via a simple cosine similarity search against a dictionary of prototypes. This noise tolerance mimics the fault tolerance characteristic of biological brains and ensures the stability of 'X' during the chaotic sensory influx of real-world environments.   

## 4. The Dynamics of Impulsion: Conceptors and the Javana Phase

While VSAs represent the static algebraic composition of 'X' at a single discrete moment, cognition is inherently temporal. The Abhidhamma dictates that active semantic processing occurs through a sequence of highly recurrent states, specifically localized during the seven moments of the _Javana_ (impulsion) phase. To govern the dynamics of this recurrent phase and enable true continuous learning, the architecture integrates Conceptors, a neuro-computational mechanism introduced by Herbert Jaeger for controlling Recurrent Neural Networks (RNNs).   

When an RNN is driven by a specific temporal pattern, its internal excited states do not scatter randomly; they form a distinct geometric cloud within the high-dimensional state space. A Conceptor is a symmetric positive semidefinite matrix that functions as a soft projection operator, encapsulating the ellipsoidal envelope of this state cloud. For a collection of neural activation vectors X=[x1​,x2​,...,xn​] representing a specific cognitive concept, the state correlation matrix is defined as R=n1​XX⊤. The Conceptor matrix C is then calculated through a closed-form solution that minimizes reconstruction loss:   

C=R(R+α−2I)−1

Here, α represents an aperture parameter that controls the resolution or precision of the concept. A large aperture causes the conceptor to approach the identity matrix, generalizing broadly, while a small aperture produces a highly narrow, specific subspace projection.   

During the seven moments of the _javana_ phase, the cognitive unit 'X' actively matches the incoming VSA hypervector against internalized conceptor matrices. Crucially, Conceptors admit a full Boolean algebra (AND, OR, NOT), allowing the system to perform complex neuro-symbolic reasoning and logic directly within the continuous dynamics of the neural network.   

These operations map directly to the Abhidhamma's description of how the mind steers its trajectory based on prior conditioning and volition. If the system perceives an ambiguous stimulus, it can dynamically evaluate competing hypotheses. The NOT operation (¬C=I−C) captures the orthogonal complement to the subspace, effectively creating a neuro-dynamical shield that suppresses a dominant but incorrect hypothesis. The AND operation (C1​∧C2​=(C1−1​+C2−1​−I)−1) projects onto components simultaneously present in two conceptors, enabling the intersection of attributes. This algebraic manipulation of dynamic patterns provides a rigorous mathematical equivalent to the Abhidhamma's assertion that _cittas_ are conditionally steered by mental factors, shaping cognitive momentum without requiring the massive retraining cycles typical of backpropagation.   

## 5. Intentionality and the Active Inference Engine

To render the cognitive unit 'X' truly autonomous, it requires an intrinsic motivational mechanism that drives learning and action. The Abhidhamma emphasizes that cognition is not passive reception; it is a process defined by conditionality and deep-seated intent (_cetana_). In modern computational neuroscience, this intentionality is modeled using the Active Inference framework, rooted in the Free Energy Principle formulated by Karl Friston.   

Active inference posits a unifying principle for perception and action: all cognitive systems maintain a generative model of their environment and operate to jointly minimize variational free energy. Free energy acts as a mathematically tractable upper bound on "surprise"—the negative log evidence of sensory data given the internal world model. Instead of relying on externally engineered reward functions (as in standard Reinforcement Learning), an active inference agent acts to fulfill its prior preferences by sampling policies that minimize Expected Free Energy (EFE).   

Within the discrete Abhidhamma pipeline, the _Votthapana_ (determining) moment serves as the exact phase where the discrepancy between the incoming sensory hypervector and the internal world model is evaluated. The system calculates the formal free energy F with respect to a recognition density Q(s) over hidden states s:   

$$ F[Q(s)] = D_{KL}[Q(s) |

| P(s)] - \mathbb{E}_{Q(s)}[\log P(o | s)] $$

where DKL​ is the Kullback–Leibler divergence, P(s) is the prior, and P(o∣s) is the generative model likelihood. If the free energy is near zero, indicating perfect prediction, the system may bypass deep processing and return to the _bhavanga_ state, conserving energy.   

However, if the surprise is high, the sequence enters the 7-moment _Javana_ phase, which functions as a recursive epistemic foraging loop. In this loop, the agent evaluates potential internal state transitions or policies (π) to minimize the Expected Free Energy G:   

G(π)=EQ(oτ​,sτ​∣π)​[logQ(sτ​∣π)−logP(oτ​,sτ​)]

The _Javana_ loop recurrently projects the Conceptor matrix representing the current state forward in time, simulating counterfactual outcomes. The policies are scored by the sum of expected epistemic value (actions that resolve uncertainty and gather information) and pragmatic value (actions that align outcomes with internal prior preferences). This firmly grounds the pre-linguistic unit 'X' in a self-supervised cycle of continuous predictive updating, mapping the philosophical concept of _kamma_ (volitional action) to the rigorous optimization of a forward-looking generative model.   

## 6. Continuous Lifelong Learning: Overcoming Catastrophic Forgetting

A critical failure of current artificial intelligence architectures is the inability to update memories continuously from an ongoing stream of sensory inputs without inducing catastrophic forgetting. When a standard artificial neural network is sequentially fine-tuned on new tasks, the gradient-descent updates inadvertently destroy the precise weight configurations that encoded prior knowledge.   

Biological brains avoid this limitation through a dual-memory architecture involving corticohippocampal circuits, known as the Complementary Learning Systems (CLS) theory. In this biological model, the hippocampus acts as a rapid, specific memory encoder that captures the immediate details of episodic events. Concurrently, the neocortex acts as a slow, generalized knowledge integrator, extracting structural patterns over time and periodically consolidating memories during sleep via spontaneous replay. At the localized synaptic level, biological neurons exhibit metaplasticity—the threshold for future synaptic changes shifts based on the synapse's prior learning history. Large synaptic spines that have stored substantial amounts of critical memory learn at a slower rate, protecting established knowledge while allowing new synapses to remain highly adaptive.   

### 6.1 Spiking Neural Networks and Gradient-Free Metaplasticity

To replicate this lifelong learning capacity within the architecture of 'X', the underlying computational substrate utilizes Neuromorphic Spiking Neural Networks (SNNs) rather than traditional dense artificial neural networks. SNNs communicate via asynchronous, discrete binary spikes, closely mimicking biological neurons and inherently supporting event-driven temporal sparsity.   

Crucially, learning in SNNs does not require computationally expensive global backpropagation. Instead, it relies on local, bio-plausible rules such as Spike-Timing-Dependent Plasticity (STDP), which adjusts the connection strength between two neurons based strictly on the precise relative timing of their individual spikes. To actively combat catastrophic forgetting, this architecture implements an advanced gradient-free metaplasticity mechanism derived from the Inter-Spike Interval Coefficient of Variation (ISI-CV).   

In neuroscience, the ISI-CV is a well-established measure of neuronal firing regularity. Neurons that fire with a regular, clock-like rhythm (low CV) are identified as encoding stable, task-critical features (such as foundational Image Schemas). The system utilizes this local spike timing data to infer synaptic importance. Synapses attached to low-CV neurons are effectively "frozen" or highly regularized during subsequent learning phases, similar to Elastic Weight Consolidation (EWC), but achieved entirely through local, on-chip spike monitoring without gradient calculations. Conversely, neurons with highly irregular, noisy firing patterns (high CV) are permitted to adapt freely, allowing them to rapidly encode novel stimuli without overwriting the network's structural foundation.   

### 6.2 The Tadarammana Phase: Sparse and Localized Consolidation

Within the rigid Abhidhamma 17-moment sequence, the final two moments constitute the _Tadarammana_ (registration) phase. This phenomenological phase perfectly aligns with the architectural requirement for localized, sparse memory consolidation.   

Rather than updating the entire synaptic network continuously during active perception, the cognitive engine temporarily stores the conceptual modifications generated during the _Javana_ phase within a localized, high-dimensional VSA working memory hypervector. If the Active Inference module determines that the sequence carries sufficient epistemic value—meaning it significantly reduced free energy, resolved an ambiguity, or encountered a meaningful environmental anomaly—the _Tadarammana_ phase triggers a structural synaptic update.   

By restricting weight updates specifically and exclusively to the 16th and 17th moments of the cognitive sequence, the system effectively shields its foundational structural memory from the continuous "noise pollution" of irrelevant background stimuli. This localized "Karmic Registration" isolates the fluid working memory (the _Javana_ processing) from the long-term memory (the synaptic substrate), achieving true lifelong learning without the need for massive, computationally draining offline replay buffers.   

## 7. The 17-Moment Cognitive Engine: An Architectural Synthesis

Integrating these disparate theories—the Abhidhamma _cittavithi_, Conceptual Spaces, Vector Symbolic Architectures, Conceptors, Active Inference, and Neuromorphic SNNs—yields a comprehensive blueprint for a highly efficient, autonomous cognitive engine. This framework enables the processing of the pre-linguistic unit 'X' through a cycle of continuous intelligence generation. The viability of this approach has been empirically validated in advanced neuromorphic paradigms, such as the Sparse Event-Driven Sequence Processing (SEDSP) engine, which directly translates the 17-moment cycle into hardware execution.   

The comprehensive architecture of 'X' unfolds as follows, mapping the ancient psychological model to modern computational mechanics:

|Processing Phase|Abhidhamma Moments|Computational Equivalent|Functional Mechanism|
|---|---|---|---|
|**I. Baseline**|1-3. Past, Vibrating, and Arrested _Bhavanga_|**Asynchronous Event Detection**|The system rests in an ultra-low-power neuromorphic state. An anomalous sensory spike train interrupts the baseline SNN firing pattern, ending the passive cycle.|
|**II. Apprehension**|4-5. Sense-Door Adverting & Sense Consciousness|**Feature Extraction & Image Schema Grounding**|The SNN extracts spatio-temporal features, mapping them into geometric Conceptual Spaces (enforcing convexity) to identify pre-linguistic Image Schemas.|
|**III. Assimilation**|6-7. Receiving & Investigating|**VSA Hypervector Binding**|The raw Image Schema is bound (⊗) with relevant spatial, temporal, and modal context vectors, constructing a composite high-dimensional state representing the unit 'X'.|
|**IV. Determination**|8. Determining (_Votthapana_)|**Free Energy Evaluation & Salience Gating**|The system calculates the divergence between the incoming hypervector 'X' and its internal World Model. High expected free energy dictates that the stimulus is highly salient, opening the gate for active processing.|
|**V. Impulsion**|9-15. _Javana_ (7 Moments)|**Conceptor Dynamics & Active Inference Loop**|A 7-step recursive evaluation utilizing Conceptors. The system applies Boolean logic to internal sub-spaces to test hypotheses, simulate counterfactual policies, and minimize uncertainty. Semantic evaluation and decision-making occur here.|
|**VI. Consolidation**|16-17. Registration (_Tadarammana_)|**STDP / ISI-CV Metaplasticity Update**|"Karmic Registration." If the _Javana_ phase produced a high-value insight, the result is structurally embedded into the SNN via localized, gradient-free synaptic plasticity, ensuring continual learning without catastrophic forgetting.|

  

## 8. Achieving Ultra-Low Computational Resource Consumption

The traditional deep learning paradigm relies on dense matrix multiplications executed across deep layers of static graphs. This approach forces the hardware to expend massive amounts of energy to process every input frame equally, irrespective of the informational value or salience of the data. The Abhidhamma-inspired architecture fundamentally resolves this inefficiency, achieving true energy proportionality through three compounding mechanisms:   

First, by utilizing **Temporal Elasticity and Event-Driven Sparsity**. Because cognition is modeled as a sequence of discrete moments triggered only by specific threshold events (the interruption of the _bhavanga_), the system remains entirely dormant when observing static or highly predictable environments. The underlying Spiking Neural Networks operate such that energy is only consumed when an individual neuron spikes, closely approximating the extraordinary 20-watt power budget of the biological human brain.   

Second, through **Proactive Salience-Driven Gating**. The 8th moment of the sequence (_Votthapana_) acts as an absolute, uncompromising gatekeeper. If an input aligns perfectly with the predictive world model (yielding zero free energy or surprise), the cognitive sequence terminates early, immediately dropping back into the _bhavanga_ resting state. The computationally intensive _Javana_ loop is instantiated strictly for novel, complex, or high-priority inputs.   

Third, by employing **Recursive "Javana" Execution over Deep Feedforward Stacks**. Instead of passing data through hundreds of sequential neural network layers to achieve depth of processing, the architecture utilizes a highly compact, fixed 7-step recurrent loop. By leveraging Conceptor algebra to iteratively refine interpretations within this localized loop, the network size remains remarkably small. This temporal recurrence extracts depth and semantic nuance while dramatically reducing the memory bandwidth and parameter count required on the silicon hardware.   

Recent empirical evaluations of analogous frameworks, such as the SEDSP engine deployed on neuromorphic hardware, have demonstrated that this combination of temporal elasticity and localized, salience-driven gating can yield over a 138-fold reduction in total energy consumption in sparse, real-world edge environments, compared to state-of-the-art dense transformer architectures.   

## Conclusion: The Horizon of Neuro-Symbolic Cognition

By abandoning the continuous, gradient-heavy models of contemporary connectionism in favor of the highly discrete, moment-by-moment phenomenological framework of the Buddhist Abhidhamma, we unlock a revolutionary paradigm for artificial intelligence.

The pre-linguistic cognitive unit 'X' is not a static, arbitrary token of Mentalese, nor is it a meaningless, unmoored continuous vector. It is a highly structured, dynamically bound hypervector grounded in the physical reality of Image Schemas. It arises conditionally, operates within the mathematically rigorous confines of Conceptual Spaces and Conceptor algebra to actively minimize free energy, and perishes, leaving behind only the distilled, highly regulated synaptic traces of its passage.   

This architecture elegantly bridges the long-standing gap between deep neural perception and high-level symbolic reasoning. By constraining continuous learning to the highly localized _Tadarammana_ phase and utilizing the biological principles of metaplasticity and inter-spike intervals, the system effectively conquers the persistent challenge of catastrophic forgetting. The resulting cognitive engine is uniquely suited for edge intelligence—capable of continuous, lifelong intellectual evolution while remaining firmly within the strict power and memory constraints required by deployment in the physical world. Through the rigorous mathematical fusion of ancient phenomenological epistemology and cutting-edge computational neuroscience, we achieve a structurally authentic, sustainable architecture of mind capable of autonomous transcendence.   

[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FLanguage_of_thought_hypothesis)

en.wikipedia.org

Language of thought hypothesis - Wikipedia

Opens in a new window](https://en.wikipedia.org/wiki/Language_of_thought_hypothesis)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fplato.stanford.edu%2Fentries%2Flanguage-thought%2F)

plato.stanford.edu

The Language of Thought Hypothesis (Stanford Encyclopedia of Philosophy)

Opens in a new window](https://plato.stanford.edu/entries/language-thought/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fiep.utm.edu%2Flot-hypo%2F)

iep.utm.edu

Language of Thought Hypothesis | Internet Encyclopedia of Philosophy

Opens in a new window](https://iep.utm.edu/lot-hypo/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fmedium.com%2F%40dominic.timothy%2Fthe-language-of-thought-0a53721db0ad)

medium.com

The Language of Thought. Jerry Fodor's Unspoken Code of… | by Dominic Timothy | Medium

Opens in a new window](https://medium.com/@dominic.timothy/the-language-of-thought-0a53721db0ad)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fportal.research.lu.se%2Fen%2Fpublications%2Fconceptual-spaces-the-geometry-of-thought%2F)

portal.research.lu.se

Conceptual spaces : the geometry of thought - Lund University Research Portal

Opens in a new window](https://portal.research.lu.se/en/publications/conceptual-spaces-the-geometry-of-thought/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fbooks.google.com%2Fbooks%2Fabout%2FConceptual_Spaces.html%3Fid%3DZOKMEAAAQBAJ%26source%3Dkp_book_description)

books.google.com

Opens in a new window](https://books.google.com/books/about/Conceptual_Spaces.html?id=ZOKMEAAAQBAJ&source=kp_book_description)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.mdpi.com%2F1424-8220%2F26%2F9%2F2656)

mdpi.com

Spiking Neural Networks with Continual Learning for Steering Angle Regression: A Sustainable AI Perspective - MDPI

Opens in a new window](https://www.mdpi.com/1424-8220/26/9/2656)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2604.16496v1)

arxiv.org

Gradient-Free Continual Learning in Spiking Neural Networks via Inter-Spike Interval Regularization - arXiv

Opens in a new window](https://arxiv.org/html/2604.16496v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC11788432%2F)

pmc.ncbi.nlm.nih.gov

Hybrid neural networks for continual learning inspired by corticohippocampal circuits - PMC

Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC11788432/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.archotec.ai%2Fwhitepaper)

archotec.ai

Whitepaper | Archotec AI - Autonomous Cognitive Architecture

Opens in a new window](https://www.archotec.ai/whitepaper)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2502.11380v1)

arxiv.org

Exploring the Small World of Word Embeddings: A Comparative Study on Conceptual Spaces from LLMs of Different Scales - arXiv

Opens in a new window](https://arxiv.org/html/2502.11380v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpsychologyfanatic.com%2Fbarsalous-perceptual-symbol-theory%2F)

psychologyfanatic.com

Barsalou's Perceptual Symbol Theory Explained - Psychology Fanatic

Opens in a new window](https://psychologyfanatic.com/barsalous-perceptual-symbol-theory/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=http%3A%2F%2Fruccs.rutgers.edu%2Fimages%2Fpersonal-zenon-pylyshyn%2Fclass-info%2FFP2012%2FFP2012_readings%2FBarsalou_BBS1999.pdf)

ruccs.rutgers.edu

Perceptual symbol systems - Rutgers Center for Cognitive Science

Opens in a new window](http://ruccs.rutgers.edu/images/personal-zenon-pylyshyn/class-info/FP2012/FP2012_readings/Barsalou_BBS1999.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.accesstoinsight.org%2Flib%2Fauthors%2Fmendis%2Fwheel322.html)

accesstoinsight.org

The Abhidhamma in Practice - Access to Insight

Opens in a new window](https://www.accesstoinsight.org/lib/authors/mendis/wheel322.html)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fbuddho.org%2Fthe-cognitive-process-according-to-the-abhidhamma%2F)

buddho.org

The Cognitive Process According to the Abhidhamma - Buddho.org

Opens in a new window](https://buddho.org/the-cognitive-process-according-to-the-abhidhamma/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=http%3A%2F%2Fwww.abhidhamma.com%2Ftxt_INTRODUCTION_ABHIDHAMMA_Rewata_Dhamma.pdf)

abhidhamma.com

INTRODUCTION TO THE ABHIDHAMMA

Opens in a new window](http://www.abhidhamma.com/txt_INTRODUCTION_ABHIDHAMMA_Rewata_Dhamma.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fshs.cairn.info%2Farticle%2FE_RIP_253_0341%2Fpdf%3Flang%3Den)

shs.cairn.info

From the Buddha's Teaching to the Abhidhamma - Cairn

Opens in a new window](https://shs.cairn.info/article/E_RIP_253_0341/pdf?lang=en)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FPerformance-evaluations-of-hybrid-plasticity-spiking-neural-networks-a-b-Performance_fig5_357716951)

researchgate.net

Performance evaluations of hybrid plasticity spiking neural networks a,... - ResearchGate

Opens in a new window](https://www.researchgate.net/figure/Performance-evaluations-of-hybrid-plasticity-spiking-neural-networks-a-b-Performance_fig5_357716951)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FThe-hybrid-plasticity-approach-can-effectively-ensure-the-convergence-and-accuracy-of-the_fig1_357716951)

researchgate.net

The hybrid plasticity approach can effectively ensure the convergence... | Download Scientific Diagram - ResearchGate

Opens in a new window](https://www.researchgate.net/figure/The-hybrid-plasticity-approach-can-effectively-ensure-the-convergence-and-accuracy-of-the_fig1_357716951)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.bps.lk%2Folib%2Fwh%2Fwh322_Mendis_Abhidhamma_In_Practice.pdf)

bps.lk

Wh 322/323: The Abhidhamma in Practice - Buddhist Publication Society

Opens in a new window](https://www.bps.lk/olib/wh/wh322_Mendis_Abhidhamma_In_Practice.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fmedium.com%2F%40dileepadigitalworks%2Fthe-architecture-of-transcendence-where-quantum-physics-meets-abhidhamma-and-sacred-traditions-in-f8deb778cec4)

medium.com

The Architecture of Transcendence: Where Quantum Physics Meets Abhidhamma and Sacred Traditions in the Study of Altered States | by Dileepa Pothuhera | Medium

Opens in a new window](https://medium.com/@dileepadigitalworks/the-architecture-of-transcendence-where-quantum-physics-meets-abhidhamma-and-sacred-traditions-in-f8deb778cec4)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Frukkhamula.wordpress.com%2Fabhidhamma-tutorials%2Fcitta%2F)

rukkhamula.wordpress.com

Lesson 1 – Citta - rukkha mūla - WordPress.com

Opens in a new window](https://rukkhamula.wordpress.com/abhidhamma-tutorials/citta/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fhu.kln.ac.lk%2Fdepts%2Fpali%2Fimages%2Fsarada_xv_i%2Fsarada_xv_ii%2Fsarada_xv_ii_amunudowe_hemasiri.pdf)

hu.kln.ac.lk

An Analysis of the Mind in the Theravāda Abhidhamma Piṭaka - Faculty of Humanities

Opens in a new window](https://hu.kln.ac.lk/depts/pali/images/sarada_xv_i/sarada_xv_ii/sarada_xv_ii_amunudowe_hemasiri.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.humanitiesjournal.net%2Farchives%2F2026%2Fvol8issue1%2FPartB%2F8-1-14-986.pdf)

humanitiesjournal.net

Momentariness, representation, and agency: Abhidhamma and Buddhist Epistemology in dialogue with contemporary cognitive science - International Journal of Humanities and Education Research

Opens in a new window](https://www.humanitiesjournal.net/archives/2026/vol8issue1/PartB/8-1-14-986.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.reddit.com%2Fr%2Ftheravada%2Fcomments%2F1ngevdl%2Fa_rough_qa_of_abhidhamma_with_western_ideas%2F)

reddit.com

A rough Q&A of Abhidhamma with Western ideas : r/theravada - Reddit

Opens in a new window](https://www.reddit.com/r/theravada/comments/1ngevdl/a_rough_qa_of_abhidhamma_with_western_ideas/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F397427972_The_Ephemeral_Projector_-_A_Comparative_Analysis_of_Quantum_Mechanics_and_Abhidhamma_Re-evaluating_Observer_Reality_Causality_and_Consciousness)

researchgate.net

The Ephemeral Projector - A Comparative Analysis of Quantum Mechanics and Abhidhamma. Re-evaluating Observer, Reality, Causality, and Consciousness - ResearchGate

Opens in a new window](https://www.researchgate.net/publication/397427972_The_Ephemeral_Projector_-_A_Comparative_Analysis_of_Quantum_Mechanics_and_Abhidhamma_Re-evaluating_Observer_Reality_Causality_and_Consciousness)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Faddschool.org%2Ffile%2F132687)

addschool.org

Senior part 2 Abhidhamma - Athula Dassana Dhamma School

Opens in a new window](https://addschool.org/file/132687)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fso09.tci-thaijo.org%2Findex.php%2Fjibs%2Farticle%2Fview%2F8785)

so09.tci-thaijo.org

Attentional Capture and Ethical Responsibility in Theravāda Abhidhamma: A Formal Reconstruction of the Cognitive Sequence | Journal of International Buddhist Studies - ThaiJO

Opens in a new window](https://so09.tci-thaijo.org/index.php/jibs/article/view/8785)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FBuilding-on-the-architectural-framework-of-the-Abhidhamma-Citta-process_fig1_403135633)

researchgate.net

Building on the architectural framework of the Abhidhamma Citta process. - ResearchGate

Opens in a new window](https://www.researchgate.net/figure/Building-on-the-architectural-framework-of-the-Abhidhamma-Citta-process_fig1_403135633)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fdirect.mit.edu%2Fbooks%2Fmonograph%2F2532%2Fbookpreview-pdf%2F2444788)

direct.mit.edu

CONCEPTUAL SPACES - PETER GäRDENFORS - MIT Press Direct

Opens in a new window](https://direct.mit.edu/books/monograph/2532/bookpreview-pdf/2444788)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fpdf%2F2605.28387)

arxiv.org

CLANE: Continual Learning of Actions on Neuromorphic Hardware from Event Cameras - arXiv

Opens in a new window](https://arxiv.org/pdf/2605.28387)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2605.09485v1)

arxiv.org

Semasia: A Large-Scale Dataset of Semantically Structured Latent Representations - arXiv

Opens in a new window](https://arxiv.org/html/2605.09485v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fcis.temple.edu%2Ftagit%2Fpresentations%2FPerceptual%2520Symbol%2520Systems.pdf)

cis.temple.edu

Perceptual Symbol Systems - Temple CIS

Opens in a new window](https://cis.temple.edu/tagit/presentations/Perceptual%20Symbol%20Systems.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fscispace.com%2Fpapers%2Fperceptual-symbol-systems-4spsk97r61)

scispace.com

(PDF) Perceptual symbol systems. (1999) | Lawrence W. Barsalou | 6184 Citations

Opens in a new window](https://scispace.com/papers/perceptual-symbol-systems-4spsk97r61)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F288562955_Image_Schemas)

researchgate.net

(PDF) Image Schemas - ResearchGate

Opens in a new window](https://www.researchgate.net/publication/288562955_Image_Schemas)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fapps.dtic.mil%2Fsti%2Fpdfs%2FADA458943.pdf)

apps.dtic.mil

An Image Schema Language - DTIC

Opens in a new window](https://apps.dtic.mil/sti/pdfs/ADA458943.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FImage_schema)

en.wikipedia.org

Image schema - Wikipedia

Opens in a new window](https://en.wikipedia.org/wiki/Image_schema)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2402.00956v1)

arxiv.org

Exploring Spatial Schema Intuitions in Large Language and Vision Models - arXiv

Opens in a new window](https://arxiv.org/html/2402.00956v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fdagmargromann.com%2Ffiles%2FACS_final_2017.pdf)

dagmargromann.com

Kinesthetic Mind Reader: A Method to Identify Image Schemas in Natural Language - Dagmar Gromann

Opens in a new window](https://dagmargromann.com/files/ACS_final_2017.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.ifaamas.org%2FProceedings%2Faamas2025%2Fpdfs%2Fp2875.pdf)

ifaamas.org

Grounding Agent Reasoning in Image Schemas: A Neurosymbolic Approach to Embodied Cognition - IFAAMAS

Opens in a new window](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2875.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FConceptual_space)

en.wikipedia.org

Conceptual space - Wikipedia

Opens in a new window](https://en.wikipedia.org/wiki/Conceptual_space)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC11792772%2F)

pmc.ncbi.nlm.nih.gov

The Geometry and Dynamics of Meaning - PMC - NIH

Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC11792772/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fceur-ws.org%2FVol-2003%2FNeSy17_paper1.pdf)

ceur-ws.org

Towards Grounding Conceptual Spaces in Neural Representations - CEUR-WS.org

Opens in a new window](https://ceur-ws.org/Vol-2003/NeSy17_paper1.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F220637816_Conceptual_Spaces_for_Computer_Vision_Representations)

researchgate.net

(PDF) Conceptual Spaces for Computer Vision Representations - ResearchGate

Opens in a new window](https://www.researchgate.net/publication/220637816_Conceptual_Spaces_for_Computer_Vision_Representations)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.cl.cam.ac.uk%2F%7Eaac10%2FR207%2Fdouwe_handout.pdf)

cl.cam.ac.uk

Semantics, Conceptual Spaces and the Meeting of Minds

Opens in a new window](https://www.cl.cam.ac.uk/~aac10/R207/douwe_handout.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fdtu.dk%2Fenglish%2Fnewsarchive%2F2025%2F07%2Fpeeking-inside-ai-brains-machines-learn-like-us)

dtu.dk

Peeking inside AI brains: Machines learn like us - DTU

Opens in a new window](https://dtu.dk/english/newsarchive/2025/07/peeking-inside-ai-brains-machines-learn-like-us)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.preprints.org%2Fmanuscript%2F202510.0117)

preprints.org

Designing Vector-Symbolic Architectures for Biomedical Applications: Ten Tips and Common Pitfalls - Preprints.org

Opens in a new window](https://www.preprints.org/manuscript/202510.0117)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3D9Sj1NCqT8U)

openreview.net

On convex decision regions in deep network representations - OpenReview

Opens in a new window](https://openreview.net/forum?id=9Sj1NCqT8U)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DyMMIWHbjWS)

openreview.net

On convex decision regions in deep network representations - OpenReview

Opens in a new window](https://openreview.net/forum?id=yMMIWHbjWS)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.hyperdimensionalcomputing.ai%2Fhdc-intro%2Fposts%2Fhdc-intro%2F)

hyperdimensionalcomputing.ai

Introduction to Hyperdimensional Computing

Opens in a new window](https://www.hyperdimensionalcomputing.ai/hdc-intro/posts/hdc-intro/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F330926381_Vector_Symbolic_Architectures_and_their_Applications_Computing_with_Random_Vectors_in_a_Hyperdimensional_Space)

researchgate.net

(PDF) Vector Symbolic Architectures and their Applications: Computing with Random Vectors in a Hyperdimensional Space - ResearchGate

Opens in a new window](https://www.researchgate.net/publication/330926381_Vector_Symbolic_Architectures_and_their_Applications_Computing_with_Random_Vectors_in_a_Hyperdimensional_Space)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fblog.stackademic.com%2Fneuro-symbolic-vector-symbolic-reasoning-architectures-a88c9b0239a7)

blog.stackademic.com

Neuro-Symbolic & Vector-Symbolic Reasoning Architectures | by RAKTIM SINGH

Opens in a new window](https://blog.stackademic.com/neuro-symbolic-vector-symbolic-reasoning-architectures-a88c9b0239a7)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FHyperdimensional_computing)

en.wikipedia.org

Hyperdimensional computing - Wikipedia

Opens in a new window](https://en.wikipedia.org/wiki/Hyperdimensional_computing)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.quantamagazine.org%2Fa-new-approach-to-computation-reimagines-artificial-intelligence-20230413%2F)

quantamagazine.org

A New Approach to Computation Reimagines Artificial Intelligence | Quanta Magazine

Opens in a new window](https://www.quantamagazine.org/a-new-approach-to-computation-reimagines-artificial-intelligence-20230413/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2501.05368v2)

arxiv.org

Developing a Foundation of Vector Symbolic Architectures Using Category Theory - arXiv

Opens in a new window](https://arxiv.org/html/2501.05368v2)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fmedium.com%2Flatinxinai%2Fhyperdimensional-computing-taking-ai-to-the-next-level-by-emulating-the-brain-a79286581ca1)

medium.com

Hyperdimensional Computing: Taking AI to the Next Level by Emulating the Brain - Medium

Opens in a new window](https://medium.com/latinxinai/hyperdimensional-computing-taking-ai-to-the-next-level-by-emulating-the-brain-a79286581ca1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fabs%2F1403.3369)

arxiv.org

[1403.3369] Controlling Recurrent Neural Networks by Conceptors - arXiv

Opens in a new window](https://arxiv.org/abs/1403.3369)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1406.2671)

arxiv.org

Conceptors - arXiv

Opens in a new window](https://arxiv.org/pdf/1406.2671)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.semanticscholar.org%2Fpaper%2FControlling-Recurrent-Neural-Networks-by-Conceptors-Jaeger%2Feb2b3d8b76355700a0cd2bdbb347a5b1ecd362ab)

semanticscholar.org

Controlling Recurrent Neural Networks by Conceptors - Semantic Scholar

Opens in a new window](https://www.semanticscholar.org/paper/Controlling-Recurrent-Neural-Networks-by-Conceptors-Jaeger/eb2b3d8b76355700a0cd2bdbb347a5b1ecd362ab)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F263012118_Conceptors_an_easy_introduction)

researchgate.net

(PDF) Conceptors: an easy introduction - ResearchGate

Opens in a new window](https://www.researchgate.net/publication/263012118_Conceptors_an_easy_introduction)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.emergentmind.com%2Ftopics%2Fboolean-operations-on-conceptors)

emergentmind.com

Boolean Operations on Conceptors - Emergent Mind

Opens in a new window](https://www.emergentmind.com/topics/boolean-operations-on-conceptors)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=http%3A%2F%2Fwadt18.cs.rhul.ac.uk%2Fsubmissions%2FWADT18A22.pdf)

wadt18.cs.rhul.ac.uk

A fuzzy institution for neural conceptors

Opens in a new window](http://wadt18.cs.rhul.ac.uk/submissions/WADT18A22.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fpdf%2F2605.04980)

arxiv.org

Conceptors for Semantic Steering - arXiv

Opens in a new window](https://arxiv.org/pdf/2605.04980)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fblog.gazzurelli.com%2Ffrom-predictive-coding-to-digital-brain-a-cognitive-architecture-for-ai-agents-with-persistent-f1726baf97f6)

blog.gazzurelli.com

From Predictive Coding to Digital Brain: A Cognitive Architecture for AI Agents with Persistent Memory (Part 1) - Matteo Gazzurelli

Opens in a new window](https://blog.gazzurelli.com/from-predictive-coding-to-digital-brain-a-cognitive-architecture-for-ai-agents-with-persistent-f1726baf97f6)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.emergentmind.com%2Ftopics%2Fpredictive-processing-and-active-inference)

emergentmind.com

Predictive Processing & Active Inference - Emergent Mind

Opens in a new window](https://www.emergentmind.com/topics/predictive-processing-and-active-inference)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.frontiersin.org%2Fjournals%2Fcomputational-neuroscience%2Farticles%2F10.3389%2Ffncom.2020.574372%2Ffull)

frontiersin.org

Learning Generative State Space Models for Active Inference - Frontiers

Opens in a new window](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2020.574372/full)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpar.nsf.gov%2Fservlets%2Fpurl%2F10441195)

par.nsf.gov

World Model Learning from Demonstrations with Active Inference: Application to Driving Behavior

Opens in a new window](https://par.nsf.gov/servlets/purl/10441195)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC12840411%2F)

pmc.ncbi.nlm.nih.gov

Decision, Inference, and Information: Formal Equivalences Under Active Inference - PMC

Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC12840411/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2604.23278v1)

arxiv.org

Active Inference: A Method for Phenotyping Agency in AI Systems? - arXiv

Opens in a new window](https://arxiv.org/html/2604.23278v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.frontiersin.org%2Fjournals%2Fneuroscience%2Farticles%2F10.3389%2Ffnins.2025.1768235%2Ffull)

frontiersin.org

Astrocyte-gated multi-timescale plasticity for online continual learning in deep spiking neural networks - Frontiers

Opens in a new window](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1768235/full)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fabs%2F2602.12236)

arxiv.org

[2602.12236] Energy-Aware Spike Budgeting for Continual Learning in Spiking Neural Networks for Neuromorphic Vision - arXiv

Opens in a new window](https://arxiv.org/abs/2602.12236)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC10194827%2F)

pmc.ncbi.nlm.nih.gov

A survey and perspective on neuromorphic continual learning systems - PMC

Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC10194827/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F8012657_Cascade_Models_of_Synaptically_Stored_Memories)

researchgate.net

Cascade Models of Synaptically Stored Memories | Request PDF - ResearchGate

Opens in a new window](https://www.researchgate.net/publication/8012657_Cascade_Models_of_Synaptically_Stored_Memories)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fscientific-contributions%2FSen-Lu-2166606784)

researchgate.net

Sen Lu's research works | Pennsylvania State University and other places - ResearchGate

Opens in a new window](https://www.researchgate.net/scientific-contributions/Sen-Lu-2166606784)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.sandamirskaya.eu%2Fresources%2FInteractive_Continual_Learning_for_Robots__Neuromorphic_Approach__ICONS_.pdf)

sandamirskaya.eu

Interactive continual learning for robots: a neuromorphic approach - Yulia Sandamirskaya

Opens in a new window](https://www.sandamirskaya.eu/resources/Interactive_Continual_Learning_for_Robots__Neuromorphic_Approach__ICONS_.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.research-collection.ethz.ch%2Fitems%2Fcf532b4c-d203-4d69-b821-f0e97462beb2)

research-collection.ethz.ch

Neuro-vector-symbolic architectures: Exploring computation in superposition for perception, reasoning, and combinatorial search - ETH Zurich Research Collection

Opens in a new window](https://www.research-collection.ethz.ch/items/cf532b4c-d203-4d69-b821-f0e97462beb2)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.research-collection.ethz.ch%2Fbitstreams%2F5c0600c9-ebf1-4ad1-81a3-7928c604f61d%2Fdownload)

research-collection.ethz.ch

In-memory Vector Symbolic Architectures - ETH Zurich Research Collection

Opens in a new window](https://www.research-collection.ethz.ch/bitstreams/5c0600c9-ebf1-4ad1-81a3-7928c604f61d/download)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fscientific-contributions%2FKaiyao-WU-2345232264)

researchgate.net

Kaiyao WU's research works | Shanghai University of International Business and Economics and other places - ResearchGate

Opens in a new window](https://www.researchgate.net/scientific-contributions/Kaiyao-WU-2345232264)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FNeuro-symbolic_AI)

en.wikipedia.org

Neuro-symbolic AI - Wikipedia

Opens in a new window](https://en.wikipedia.org/wiki/Neuro-symbolic_AI)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.mdpi.com%2F2079-9292%2F15%2F5%2F963)

mdpi.com

Bridging Cognitive and Expression Spaces in Creative AI by Integrating DIKWP-TRIZ and Semantic Mathematics - MDPI

Opens in a new window](https://www.mdpi.com/2079-9292/15/5/963)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.emergentmind.com%2Ftopics%2Fhybrid-cognitive-architectures)

emergentmind.com

Hybrid Cognitive Architectures - Emergent Mind

Opens in a new window](https://www.emergentmind.com/topics/hybrid-cognitive-architectures)

[

![](https://t3.gstatic.com/faviconV2?url=https://hu.kln.ac.lk/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://hu.kln.ac.lk/depts/pali/images/sarada_xv_i/sarada_xv_i_suriyawewa_wijayawimala.pdf)[

![](https://t1.gstatic.com/faviconV2?url=https://arrowriver.ca/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://arrowriver.ca/transcripts/modalities_transc.html)[

![](https://t2.gstatic.com/faviconV2?url=https://puredhamma.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://puredhamma.net/abhidhamma/essential-abhidhamma-the-basics/bhava-bhavanga-simply-explained/)[

![](https://t0.gstatic.com/faviconV2?url=https://buddhistuniversity.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://buddhistuniversity.net/content/papers/bhavanga-and-rebirth-according-to_gethin)[

![](https://t2.gstatic.com/faviconV2?url=https://puredhamma.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://puredhamma.net/forums/topic/state-of-mind-in-the-absence-of-citta-vithi-bhavanga-2/)[

![](https://t3.gstatic.com/faviconV2?url=https://pear.wpi.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://pear.wpi.edu/eventvision.html)[

![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://arxiv.org/html/2407.20633v2)[

![](https://t2.gstatic.com/faviconV2?url=https://openaccess.thecvf.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://openaccess.thecvf.com/content/CVPR2021W/EventVision/papers/Duwek_Image_Reconstruction_From_Neuromorphic_Event_Cameras_Using_Laplacian-Prediction_and_Poisson_CVPRW_2021_paper.pdf)[

![](https://t3.gstatic.com/faviconV2?url=https://i3s.unice.fr/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://i3s.unice.fr/jmartinet/sites/default/files/u47/neuromorphicstereovision.pdf)[

![](https://t3.gstatic.com/faviconV2?url=https://rpg.ifi.uzh.ch/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://rpg.ifi.uzh.ch/docs/CVPRW23_Schnider.pdf)[

![](https://t0.gstatic.com/faviconV2?url=https://neuromorphic.eecs.utk.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://neuromorphic.eecs.utk.edu/publications/2025-05-01-neuropong-the-event-based-camera-driven-embedded-neuromorphic-system/)[

![](https://t0.gstatic.com/faviconV2?url=https://www.theravada.gr/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.theravada.gr/wp-content/uploads/2021/05/Abhidhamma-in-Daily-Life.pdf)[

![](https://t0.gstatic.com/faviconV2?url=https://www.buddhistelibrary.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.buddhistelibrary.org/en/albums/asst/ebook/abhidhamma.pdf)[

![](https://t1.gstatic.com/faviconV2?url=http://abhidhamma.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](http://abhidhamma.com/Patthana_In_Daily_Life.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://www.abuddhistlibrary.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.abuddhistlibrary.com/Buddhism/B%20-%20Theravada/Teachers/Dr.%20Mehm%20Tin%20Mun/Buddha%20Abhidhamma/abhidhaultsci.pdf)[

![](https://t3.gstatic.com/faviconV2?url=https://www.themindingcentre.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.themindingcentre.org/dharmafarer/wp-content/uploads/2019/06/54.3f-Niddatandi-S-s1.16-piya.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://www.dharmawheel.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.dharmawheel.net/viewtopic.php?t=30696)[

![](https://t1.gstatic.com/faviconV2?url=https://www.namsebangdzo.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.namsebangdzo.com/Abhidhamma-Philosophy-p/12074.htm)[

![](https://t2.gstatic.com/faviconV2?url=https://www.scribd.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.scribd.com/document/93836219/Pa-Auk-The-Chariot-to-Nibbana-Part-I)[

![](https://t2.gstatic.com/faviconV2?url=https://www.scribd.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.scribd.com/document/357098912/Paticca-Samuppada-by-Rev-Mahasi-Sayadaw-Thero)[

![](https://t2.gstatic.com/faviconV2?url=https://archive.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://archive.org/download/buddhism-engl/BUDDHISM%20ENGL.rar/BUDDHISM%20ENGL%2FBHANTE%2FETC%2FOTHERS%2Fspiritual_-_buddhism_-_knowing_and_seeing.pdf)[

![](https://t1.gstatic.com/faviconV2?url=https://www.namsebangdzo.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.namsebangdzo.com/Process-of-Consciousness-and-Matter-p/9781938754623.htm)[

![](https://t2.gstatic.com/faviconV2?url=https://dokumen.pub/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://dokumen.pub/abhidhamma-papers.html)[

![](https://t0.gstatic.com/faviconV2?url=https://scholarspace.manoa.hawaii.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://scholarspace.manoa.hawaii.edu/server/api/core/bitstreams/13245add-e589-4d09-b3cd-7e8a050deaac/content)[

![](https://t2.gstatic.com/faviconV2?url=https://puredhamma.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://puredhamma.net/forums/topic/difference-between-arahant-phala-samapatti-and-nirodha-samapatti/)[

![](https://t3.gstatic.com/faviconV2?url=https://ia801408.us.archive.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://ia801408.us.archive.org/22/items/in.ernet.dli.2015.474263/2015.474263.The-Abhidhamma_text.pdf)[

![](https://t2.gstatic.com/faviconV2?url=http://www.abhidhamma.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](http://www.abhidhamma.com/txt_Manuals_of_Buddhism.pdf)[

![](https://t1.gstatic.com/faviconV2?url=http://abhidhamma.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](http://abhidhamma.com/Paramatthadipani.pdf)[

![](https://t3.gstatic.com/faviconV2?url=https://ceur-ws.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://ceur-ws.org/Vol-4080/invited1.pdf?utm_source=chatgpt.com)[

![](https://t0.gstatic.com/faviconV2?url=https://www.biorxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.biorxiv.org/content/10.64898/2025.12.09.693276v1.full-text)[

![](https://t1.gstatic.com/faviconV2?url=https://journals.plos.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013251)[

![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC12107872/)[

![](https://t3.gstatic.com/faviconV2?url=https://dspace.mit.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://dspace.mit.edu/bitstreams/d0a23a1a-ec03-4978-9ebb-c23cf0faa2df/download)[

![](https://t2.gstatic.com/faviconV2?url=https://www.techrxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177220005.52986713)

![](https://www.gstatic.com/lamda/images/immersives/google_logo_icon_2380fba942c84387f09cf.svg)

[![](https://t3.gstatic.com/faviconV2?url=https://hu.kln.ac.lk/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://hu.kln.ac.lk/depts/pali/images/sarada_xv_i/sarada_xv_i_suriyawewa_wijayawimala.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arrowriver.ca/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://arrowriver.ca/transcripts/modalities_transc.html)[![](https://t0.gstatic.com/faviconV2?url=https://www.theravada.gr/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.theravada.gr/wp-content/uploads/2021/05/Abhidhamma-in-Daily-Life.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://puredhamma.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://puredhamma.net/abhidhamma/essential-abhidhamma-the-basics/bhava-bhavanga-simply-explained/)[![](https://t0.gstatic.com/faviconV2?url=https://buddhistuniversity.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://buddhistuniversity.net/content/papers/bhavanga-and-rebirth-according-to_gethin)[![](https://t2.gstatic.com/faviconV2?url=https://puredhamma.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://puredhamma.net/forums/topic/state-of-mind-in-the-absence-of-citta-vithi-bhavanga-2/)[![](https://t3.gstatic.com/faviconV2?url=https://pear.wpi.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://pear.wpi.edu/eventvision.html)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://arxiv.org/html/2407.20633v2)[![](https://t2.gstatic.com/faviconV2?url=https://openaccess.thecvf.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://openaccess.thecvf.com/content/CVPR2021W/EventVision/papers/Duwek_Image_Reconstruction_From_Neuromorphic_Event_Cameras_Using_Laplacian-Prediction_and_Poisson_CVPRW_2021_paper.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://i3s.unice.fr/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://i3s.unice.fr/jmartinet/sites/default/files/u47/neuromorphicstereovision.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://rpg.ifi.uzh.ch/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://rpg.ifi.uzh.ch/docs/CVPRW23_Schnider.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://neuromorphic.eecs.utk.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://neuromorphic.eecs.utk.edu/publications/2025-05-01-neuropong-the-event-based-camera-driven-embedded-neuromorphic-system/)

![](https://www.gstatic.com/lamda/images/immersives/google_logo_icon_2380fba942c84387f09cf.svg)

[![](https://t0.gstatic.com/faviconV2?url=https://www.theravada.gr/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.theravada.gr/wp-content/uploads/2021/05/Abhidhamma-in-Daily-Life.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://www.buddhistelibrary.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.buddhistelibrary.org/en/albums/asst/ebook/abhidhamma.pdf)[![](https://t1.gstatic.com/faviconV2?url=http://abhidhamma.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](http://abhidhamma.com/Patthana_In_Daily_Life.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://www.abuddhistlibrary.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.abuddhistlibrary.com/Buddhism/B%20-%20Theravada/Teachers/Dr.%20Mehm%20Tin%20Mun/Buddha%20Abhidhamma/abhidhaultsci.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://www.themindingcentre.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.themindingcentre.org/dharmafarer/wp-content/uploads/2019/06/54.3f-Niddatandi-S-s1.16-piya.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://www.dharmawheel.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.dharmawheel.net/viewtopic.php?t=30696)[![](https://t1.gstatic.com/faviconV2?url=https://www.namsebangdzo.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.namsebangdzo.com/Abhidhamma-Philosophy-p/12074.htm)[![](https://t3.gstatic.com/faviconV2?url=https://ia801408.us.archive.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://ia801408.us.archive.org/22/items/in.ernet.dli.2015.474263/2015.474263.The-Abhidhamma_text.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://www.scribd.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.scribd.com/document/93836219/Pa-Auk-The-Chariot-to-Nibbana-Part-I)[![](https://t2.gstatic.com/faviconV2?url=https://www.scribd.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.scribd.com/document/357098912/Paticca-Samuppada-by-Rev-Mahasi-Sayadaw-Thero)[![](https://t2.gstatic.com/faviconV2?url=https://archive.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://archive.org/download/buddhism-engl/BUDDHISM%20ENGL.rar/BUDDHISM%20ENGL%2FBHANTE%2FETC%2FOTHERS%2Fspiritual_-_buddhism_-_knowing_and_seeing.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://www.namsebangdzo.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.namsebangdzo.com/Process-of-Consciousness-and-Matter-p/9781938754623.htm)

![](https://www.gstatic.com/lamda/images/immersives/google_logo_icon_2380fba942c84387f09cf.svg)

[![](https://t2.gstatic.com/faviconV2?url=https://dokumen.pub/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://dokumen.pub/abhidhamma-papers.html)[![](https://t0.gstatic.com/faviconV2?url=https://scholarspace.manoa.hawaii.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://scholarspace.manoa.hawaii.edu/server/api/core/bitstreams/13245add-e589-4d09-b3cd-7e8a050deaac/content)[![](https://t2.gstatic.com/faviconV2?url=https://puredhamma.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://puredhamma.net/forums/topic/difference-between-arahant-phala-samapatti-and-nirodha-samapatti/)[![](https://t3.gstatic.com/faviconV2?url=https://ia801408.us.archive.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://ia801408.us.archive.org/22/items/in.ernet.dli.2015.474263/2015.474263.The-Abhidhamma_text.pdf)[![](https://t2.gstatic.com/faviconV2?url=http://www.abhidhamma.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](http://www.abhidhamma.com/txt_Manuals_of_Buddhism.pdf)[![](https://t1.gstatic.com/faviconV2?url=http://abhidhamma.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](http://abhidhamma.com/Paramatthadipani.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://ceur-ws.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://ceur-ws.org/Vol-4080/invited1.pdf?utm_source=chatgpt.com)[![](https://t0.gstatic.com/faviconV2?url=https://www.biorxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.biorxiv.org/content/10.64898/2025.12.09.693276v1.full-text)[![](https://t1.gstatic.com/faviconV2?url=https://journals.plos.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013251)[![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://pmc.ncbi.nlm.nih.gov/articles/PMC12107872/)[![](https://t3.gstatic.com/faviconV2?url=https://dspace.mit.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://dspace.mit.edu/bitstreams/d0a23a1a-ec03-4978-9ebb-c23cf0faa2df/download)[![](https://t2.gstatic.com/faviconV2?url=https://www.techrxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177220005.52986713)