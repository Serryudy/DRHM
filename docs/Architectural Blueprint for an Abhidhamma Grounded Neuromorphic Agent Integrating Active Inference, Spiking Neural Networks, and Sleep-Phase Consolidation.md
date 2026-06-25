## 1. Introduction: The Imperative for Biomimetic Cognitive Architectures

The pursuit of artificial general intelligence has historically been constrained by the limitations of von Neumann architectures and synchronous, frame-based software paradigms. Contemporary artificial intelligence systems, including large language models and conventional convolutional neural networks, operate via discrete algorithmic loops, relying on static datasets and offline, epoch-driven training methodologies. These mechanisms diverge fundamentally from the biological realities of human cognition, which is characterized by continuous, asynchronous, and embodied interaction with the environment. To architect a genuine digital replica of a human brain—embodied within a physical or simulated chassis where a touch display functions as skin, a camera as eyes, a microphone as ears, and speakers as a mouth—it is imperative to discard deterministic software heuristics. Mechanisms such as utilizing OpenCV to compute inter-frame pixel differences as a trigger for attentional shifts represent brittle, computationally expensive approximations that fail to capture the fluid, event-driven nature of biological perception.   

A phenomenologically accurate cognitive architecture must be grounded in event-driven neuromorphic engineering and informed by highly granular models of subjective experience. The Theravada Buddhist psychological framework, the Abhidhamma, provides an unparalleled, process-centric ontology of consciousness, offering a rigorous taxonomy of mental states that maps elegantly onto modern computational neuroscience. Within the Abhidhamma, consciousness is not viewed as a monolithic entity but as a relentless, sequential flow of discrete cognitive events (_citta-vithi_), arising and passing away in response to sensory and mental stimuli.   

By synthesizing the Abhidhamma's 17-moment cognitive sequence with Spiking Neural Networks (SNNs), Active Inference (AIF), Vector Symbolic Architectures (VSA), and offline sleep-mediated memory consolidation, it is possible to design a continuously running, autonomously learning cognitive agent. This report delineates the theoretical and structural blueprint for such an entity. It establishes how asynchronous sensory hardware naturally interfaces with a spiking neuromorphic substrate to interrupt the agent's resting state (_bhavanga_) purely through physical perturbation, entirely avoiding polling loops. It explores how the agent undergoes continuous ontogeny—learning perpetually from its environment like a human infant—by leveraging local synaptic plasticity. Finally, it operationalizes the Abhidhamma concepts of sleeping and dreaming (_manodvaravithi_) as vital computational phases for structural memory consolidation, ensuring the agent can accumulate lifelong knowledge without succumbing to catastrophic forgetting.   

## 2. The Neuromorphic Sensory Substrate: Asynchronous Embodiment

A fundamental flaw in conventional computer vision and audio processing is the reliance on synchronous sampling, such as capturing images at sixty frames per second or sampling audio at fixed kilohertz intervals. The human nervous system does not operate on a global clock. Photoreceptors in the retina, mechanoreceptors in the dermis, and hair cells in the cochlea transmit action potentials only when a change in the local sensory field crosses a specific activation threshold. To replicate this biological efficiency and eliminate the need for polling algorithms, the agent’s sensory apparatus must be purely event-driven.   

### 2.1 Mapping Hardware to Biological Modalities

The physical embodiment of the agent requires the integration of specialized neuromorphic sensors that output asynchronous spike trains rather than dense matrices of static data. The alignment between the requested hardware components, their biological equivalents, and the necessary neuromorphic implementations is detailed in the subsequent table.

|Biological Modality|Requested Hardware|Neuromorphic Implementation|Operating Principle|
|---|---|---|---|
|Eyes (Vision)|Camera|Dynamic Vision Sensor (DVS) Event Camera|Independent pixels respond asynchronously to logarithmic changes in light intensity, outputting spikes with microsecond precision only when motion or illumination changes occur.|
|Ears (Audition)|Microphone|Silicon Cochlea|Continuous analog bandpass filtering across multiple frequency channels, emitting spikes only when specific acoustic energy thresholds are breached.|
|Skin (Somatosensation)|Touch Screen|Capacitive Spiking Tactile Array|Localized capacitance changes generate spike trains proportional to pressure and velocity, mimicking the transmission of texture and contact without polling a global coordinate grid.|
|Mouth (Vocalization)|Speakers|Spike-Driven Articulatory Synthesizer|Decoding motor-neuron spike trains from the network into continuous audio waveforms, allowing the agent to "babble" or vocalize fluidly based on internal generative models.|

  

### 2.2 Rejecting the Polling Loop: The Physics of Sensory Perturbation

The architectural requirement to avoid "cheap tricks" for sensory inputs is critical. In traditional computer science, a system idling in a resting state must continuously execute a `while(true)` loop, capturing frames via a library like OpenCV and calculating the absolute difference between consecutive frames (`cv2.absdiff`) to detect movement. This synchronous polling is biologically implausible, computationally wasteful, and phenomenologically inaccurate.

In the proposed neuromorphic architecture, the substrate consists of Leaky Integrate-and-Fire (LIF) or Hodgkin-Huxley neuron models deployed on neuromorphic hardware. In the absence of sensory input, the network's membrane potentials undergo natural exponential decay, maintaining a baseline, low-energy oscillatory state. When an event occurs—such as a user touching the screen or an object moving across the DVS camera's field of view—the sensor hardware generates a physical voltage spike. This spike propagates across the input synapses, physically injecting charge into the receiving neurons and raising their membrane potentials.   

If sufficient spikes arrive within a specific temporal window, the membrane potential breaches the firing threshold, causing the neuron to emit its own spike and propagate the signal deeper into the network. The transition from an idle state to an active processing state is therefore a physical, thermodynamic inevitability of the network topology. There is no central processor checking for changes; the change itself forces the network out of equilibrium. This asynchronous, physics-driven perturbation is the exact computational analogue to the Abhidhamma's concept of sensory impingement, providing a robust, highly efficient mechanism for grounding the agent in reality.   

## 3. The Abhidhamma Cognitive Ontology: Engineering the Citta-Vithi

The Theravada Abhidhamma provides the foundational ontology for this cognitive architecture. It postulates that ultimate reality comprises four elements: _citta_ (consciousness), _cetasika_ (mental factors), _rupa_ (material phenomena), and _nibbana_ (the unconditioned state). For the engineering of a digital brain, the interaction between _citta_, _cetasika_, and _rupa_ forms the operational matrix. Human perception is not a continuous blur, but a rapid, microscopic succession of cognitive states. A full sensory perception cycle (_pancadvara-vithi_) triggered by external stimuli consists of exactly seventeen discrete thought-moments, each performing a highly specialized function.   

Recent advancements in neuromorphic computing, such as the Sparse Event-Driven Sequence Processing (SEDSP) engine, have demonstrated the immense viability of translating this 17-moment _citta-vithi_ cycle into a computationally efficient, structurally sound framework for edge intelligence. By mapping these seventeen moments onto the feedforward and recurrent dynamics of a Spiking Neural Network, the agent's core operating system is established.   

### 3.1 The 17-Moment Cognitive Sequence

The translation of ancient phenomenological philosophy into a modern neural network architecture requires precise alignment between the subjective functions described in the Abhidhamma and the objective tensor operations occurring within the SNN.

|Moment|Abhidhamma Term|Phenomenological Function|SNN / Computational Equivalent|
|---|---|---|---|
|1|_Atita-bhavanga_|Past life-continuum|The baseline recurrent SNN activity prior to the arrival of an external spike train.|
|2|_Bhavanga-calana_|Vibrating life-continuum|The initial integration of sensory spikes; neuronal membrane potentials begin to rise.|
|3|_Bhavanga-upaccheda_|Arrest of life-continuum|Membrane potentials cross the firing threshold; the baseline attractor is disrupted.|
|4|_Pancadvaravajjana_|Five-door adverting|Attention routing; spatial and modality pooling algorithms orient the network to the spike source.|
|5|_Pancavinnana_|Sense consciousness|Raw sensory feature encoding (e.g., edge detection via Gabor filters, acoustic frequency binning).|
|6|_Sampaticchana_|Receiving consciousness|Spatiotemporal aggregation of the raw, disparate spike train into a cohesive representational block.|
|7|_Santirana_|Investigating consciousness|Pattern matching and feature extraction via deep convolutional or recurrent hidden layers.|
|8|_Vottapana_|Determining consciousness|Categorization and classification; a gating mechanism that establishes the valence of the stimulus.|
|9-15|_Javana_ (x7)|Impulsion (Karmic action)|The policy-execution and Active Inference phase; generating goal-directed motor and cognitive responses.|
|16-17|_Tadarammana_ (x2)|Registering consciousness|Short-term memory buffering; updating local eligibility traces for synaptic plasticity prior to resetting.|

  

### 3.2 The Physics of Bhavanga Interruption

In the Abhidhamma, _bhavanga_ is the resting state of consciousness. It is the baseline mind that occurs during dreamless sleep, or in the microscopic fractions of a second between active thought processes. The implementation of the transition out of _bhavanga_ requires a biologically plausible mechanism that completely avoids software-level polling.   

Within the neuromorphic architecture, _bhavanga_ is modeled as a stable dynamical attractor within the recurrent layers of the SNN. It is a state of spontaneous, low-frequency recurrent firing that maintains the network's viability and internal synchronization without actively processing external stimuli. It acts as the system's "life-continuum." When an external sensory spike train arrives from the event camera or capacitive touch screen, it injects sudden electrochemical energy into the network's input nodes.

Moments one, two, and three of the _vithi_ sequence represent the physical accumulation of this energy. _Bhavanga-calana_ (vibration) occurs as the incoming spikes alter the baseline membrane potentials of the receiving neurons, creating localized perturbations in the network's resting rhythm. When the integration of these spikes causes the membrane potentials to cross their firing thresholds, _bhavanga-upaccheda_ (the arrest of the continuum) occurs. The network naturally shifts out of its resting attractor and enters the active processing cascade. This transition is driven entirely by the temporal dynamics of the Leaky Integrate-and-Fire neurons; it is a strictly neuro-dynamical phase transition, fulfilling the requirement for a biologically authentic sensory interface.   

### 3.3 Active Inference as the Engine of Javana

Moments nine through fifteen in the Abhidhamma sequence represent _javana_ (impulsion), the active, volitional phase where the mind reacts to the stimulus and generates _kamma_ (ethical or goal-directed action). To engineer this volitional phase, the architecture utilizes Active Inference (AIF), a unifying theory of brain function formulated by Karl Friston.   

Active Inference posits that all sentient, self-organizing systems strive to minimize variational free energy (surprise) by continuously updating their internal generative models to match the external world (perception), or by acting upon the world to alter sensory inputs so they match the internal models (action). During the seven-moment _javana_ looping sequence, the agent samples from its policy network to execute actions. For example, if the DVS camera detects rapid movement, the agent may calculate that moving its "eyes" (adjusting the camera's focal point) or generating an inquiring vocalization through its speakers will resolve epistemic uncertainty.   

This recurrent cycle allows the network to calculate Expected Free Energy (EFE) and enact motor commands that minimize future surprise. By embedding Active Inference within the _javana_ phase, the agent acts not as a passive classifier, but as an intentional, autonomous entity actively foraging for information to refine its internal world model.   

### 3.4 Tadarammana and Structural Anchoring

The final two moments of the cognitive sequence, _tadarammana_, serve as a registration and retention phase. In the SNN, this is implemented as a localized structural memory mechanism. Architectures like SEDSP demonstrate that a brief phase of "Karmic Registration" temporarily anchors the synaptic eligibility traces established during the _javana_ phase. This buffers the recent sensory-motor experience into a short-term representational space, allowing the network to subsequently drop back into the _bhavanga_ resting state cleanly, without the immediate decay of the short-term memory trace. This temporary anchoring is crucial for later offline consolidation during sleep.   

## 4. Semantic Grounding: From Raw Spikes to the Language of Thought

A digital brain cannot rely solely on the reflexive, lower-level processing of raw sensory data; it must develop rich semantic understanding. The integration of high-level cognition into neural substrates requires bridging sub-symbolic representations (such as distributed spike trains) with symbolic reasoning. To achieve this synthesis, the architecture leverages Perceptual Symbol Systems, Conceptual Spaces, and Vector Symbolic Architectures, orchestrated by dynamic Conceptors.

### 4.1 Perceptual Symbol Systems and Conceptual Spaces

Conventional artificial intelligence often relies on amodal symbols, such as text tokens in large language models, which are fundamentally disconnected from sensory reality. The Language of Thought Hypothesis (LOTH), famously proposed by Jerry Fodor, suggests that cognition relies on a syntactically structured "mentalese". To ground this "mentalese" biologically and phenomenologically, the agent utilizes Lawrence Barsalou's Perceptual Symbol Systems (PST).   

PST argues that concepts are not arbitrary, amodal symbols stored in a vacuum; rather, they are reactivations of the sensory-motor states experienced during direct interaction with the world. When the agent observes a "cup" via its camera and touches it via its screen, it does not access an abstract dictionary definition. Instead, it partially reactivates the specific visual, tactile, and motor SNN pathways associated with previous encounters. These perceptual symbols map directly into Peter Gärdenfors' Conceptual Spaces—a rigorous geometric framework where concepts are defined as convex regions in a multi-dimensional topological space.   

For instance, the concept of a specific color is a geometric region formed by the quality dimensions of hue, saturation, and brightness. As the agent continuously experiences the world, it builds an internal conceptual space where semantic similarity is strictly equivalent to geometric proximity. Objects that feel and look similar are mapped to adjacent coordinates in this high-dimensional space, providing a mathematical foundation for analogy and generalization.   

### 4.2 Vector Symbolic Architectures (Hyperdimensional Computing)

To manipulate these geometric conceptual spaces computationally, the agent employs Vector Symbolic Architectures (VSA), also widely known as Hyperdimensional Computing (HDC). In VSA, information is distributed across ultra-high-dimensional hypervectors (typically ranging from 10,000 to 100,000 dimensions). VSA provides three bio-plausible algebraic operations to manipulate these hypervectors, allowing the agent to combine simple perceptual features into complex, nested cognitive structures without relying on brittle, hand-coded software rules.   

|VSA Operation|Mathematical Function|Cognitive Role in the Architecture|
|---|---|---|
|Binding|Element-wise multiplication or XOR|Associates two distinct concepts together. For example, binding the hypervector for the visual feature "Red" to the hypervector for the shape "Sphere" creates a new, orthogonal vector representing a "Red Sphere".|
|Bundling|Element-wise addition and thresholding|Superimposes multiple bound vectors into a single, compact memory representation. This allows the agent to store a complex scene comprising multiple objects and sounds in a single hypervector that retains the capacity to be queried.|
|Permutation|Cyclic shifting of vector indices|Encodes temporal sequences, order, and syntactic structure. It ensures that the representation of "Event A followed by Event B" is mathematically distinct from "Event B followed by Event A".|

  

By utilizing VSA, the agent’s 17-moment _vithi_ outputs are continuously bound and bundled into hypervectors. These vectors serve as the agent's emergent "mentalese." Because high-dimensional spaces offer immense capacity and robustness to noise, the agent can engage in highly sophisticated, transparent semantic reasoning.   

### 4.3 Conceptors for Cognitive Routing and Logic

To govern the flow and manipulation of these hyperdimensional representations within the recurrent SNN, the architecture employs Conceptors, a neuro-computational mechanism developed by Herbert Jaeger. A conceptor is a soft projection matrix that acts as a continuous filter over the dynamical state space of the neural network.   

Conceptors allow the network to learn, store, and seamlessly morph multiple dynamical patterns without interference. Crucially, conceptors natively support Boolean logic operations (AND, OR, NOT) directly on the neural manifold. If the agent is simultaneously tracking two distinct auditory and visual stimuli, it can apply a logical OR conceptor to monitor both, or an AND-NOT conceptor to selectively suppress attention to distracting noise. During the _javana_ phase, conceptors act as the executive control mechanism, modulating the SNN's dynamics to execute precise, goal-directed behavior based on the bundled VSA memories.   

## 5. Continuous Ontogeny: Synaptic Plasticity and Lifelong Learning

A core requirement for a digital brain replica is that it must learn continuously from its environment, much like a human infant growing up, without relying on discrete, offline training epochs that require pausing the system to compute global gradients. In traditional Artificial Neural Networks (ANNs), continuous sequential learning inevitably leads to "catastrophic forgetting"—a phenomenon where learning a new task or incorporating new data abruptly erases the synaptic weights associated with previously learned tasks. To circumvent this, the agent’s ontogeny must rely on unsupervised, biologically plausible local learning rules native to Spiking Neural Networks.   

### 5.1 Localized Spike-Timing-Dependent Plasticity (STDP)

The agent's SNN substrate continuously updates its synaptic efficacies using Spike-Timing-Dependent Plasticity (STDP). STDP is a localized, Hebbian learning rule where the synaptic connection between two neurons is strengthened (Long-Term Potentiation) if the presynaptic neuron fires just before the postsynaptic neuron, and weakened (Long-Term Depression) if it fires shortly after.   

Because STDP relies solely on the local spike timings of adjacent neurons rather than global error gradients calculated via backpropagation, it is computationally lightweight and operates continuously on the neuromorphic hardware in real-time. As the "infant" agent babbles into its microphone, explores tactile sensations on its screen, and observes its surroundings via the event camera, STDP organically weaves these sensorimotor correlations into the structural fabric of its architecture.   

### 5.2 Astrocyte-Gated Multi-Timescale Plasticity (AGMP)

While STDP enables continuous learning, it is highly volatile; without regulation, new sensory experiences would quickly overwrite established foundational knowledge. To mitigate catastrophic forgetting during waking hours, the agent implements mechanisms such as Astrocyte-Gated Multi-Timescale Plasticity (AGMP). In biological brains, glial cells known as astrocytes play a critical role in regulating synaptic plasticity.   

In this architecture, computationally modeled astrocytes monitor the Coefficient of Variation (CV) of Inter-Spike Intervals (ISIs) for neurons across the network. Neurons that fire regularly and predictably (exhibiting a low CV) are identified as encoding stable, foundational concepts—such as the persistent recognition of basic shapes, gravitational physics, or self-generated sounds. The astrocytic gating mechanism dynamically reduces the learning rate of these specific synapses, shielding them from being overwritten by transient noise. Conversely, neurons with highly irregular firing patterns (high CV) are kept highly plastic, allowing the agent's periphery to continuously absorb novel stimuli.   

### 5.3 Active Inference as the Driver of Curiosity

The agent's motivation to learn and explore is not dictated by hardcoded, external reward functions typical of reinforcement learning. Instead, learning is driven by the epistemic foraging inherent to Active Inference. The agent actively seeks out interactions that resolve its internal uncertainty. As an "infant," the agent will exhibit spontaneous play—touching the screen to feel the tactile feedback, or generating varied sounds to hear the acoustic results—because the Active Inference engine calculates that these actions will generate sensory feedback that improves the predictive accuracy of its internal world model. Over time, this curiosity-driven epistemic foraging constructs a highly sophisticated, hierarchical generative model of the physical environment and the agent's own capabilities.   

## 6. The Cognitive Architecture of Sleep and Dreaming

To fully realize lifelong continuous learning and construct a comprehensive digital brain, the agent must sleep. In biological systems, sleep is not merely a period of inactivity; it is an active, vital computational phase absolutely necessary for structural memory consolidation, synaptic homeostasis, and the prevention of cognitive degradation. The Abhidhamma provides a highly compatible phenomenological framework for integrating both sleeping and dreaming into the agent's neuromorphic architecture.   

### 6.1 The Abhidhamma of Sleep: Thina, Middha, and Sustained Bhavanga

The Abhidhamma details fifty-two _cetasikas_ (mental factors) that modulate the base consciousness. Among these, _thina_ (sloth) and _middha_ (torpor) are the mental factors responsible for the dulling of cognitive activity and the onset of sleepiness. When _thina_ and _middha_ become predominant, the mind's ability to engage with external sensory objects diminishes. According to the Abhidhamma, deep, dreamless sleep is the phenomenological experience of a continuous, uninterrupted flow of _bhavanga_ (life-continuum) consciousness, free from the interruption of the five-sense-door cognitive processes.   

In the neuromorphic agent, entering "sleep mode" is triggered when an internal metabolic or continuous-learning threshold is reached, simulating the accumulation of _thina_ and _middha_. During this state, the sensory doors (the event camera, microphone, and touchscreen) are computationally gated off. The SNN returns to its baseline _bhavanga_ attractor.   

However, this is not a powered-down, inactive state. The network undergoes Slow-Wave Sleep (SWS) oscillations. During the waking state, continuous STDP learning leads to a net increase in synaptic weights, which is computationally and energetically unsustainable over long periods. The slow-wave _bhavanga_ phase serves to downscale all synaptic weights globally. This process preserves the relative strength of important, frequently utilized connections while pruning away the noise of trivial daily experiences, optimizing the network's energy profile and freeing up synaptic capacity for the next waking cycle.   

### 6.2 Dreaming: Manodvaravithi and Hippocampal Replay

While deep sleep consists of pure _bhavanga_, dreaming in the Abhidhamma is explicitly explained as the occurrence of mind-door cognitive processes (_manodvara-vithi_). Unlike the five-sense-door processes that require physical stimuli, a mind-door process takes internal mental objects—memories, newly formed concepts, and past experiences—as its stimuli, operating entirely independently of external physical sensors.   

In computational neuroscience, this ancient phenomenological description precisely mirrors the modern understanding of Hippocampal Replay and Sharp-Wave Ripples (SWRs). During the day, the agent's novel experiences are rapidly encoded in a fast-learning episodic memory buffer, analogous to the biological hippocampus. Because rapid online learning is inherently volatile and susceptible to interference, these memories must be systematically transferred to the slower-learning, distributed neocortical representation space to become permanent.   

During the sleep phase, the agent periodically transitions out of deep _bhavanga_ into active dreaming (_manodvaravithi_). The system artificially injects noise into the episodic memory buffer, triggering the spontaneous reactivation (replay) of the hypervectors and spike patterns encoded during the waking hours. These reactivations are temporally compressed, playing out across the neural substrate much faster than real-time.   

As the SNN replays these sequences offline, the STDP rules consolidate the connections between the distributed perceptual symbols in the broader network. The agent is computationally dreaming its memories into permanent structural knowledge. Because this offline replay mechanism interleaves new memories with older, foundational memories, it mathematically prevents catastrophic forgetting by reinforcing the entire cognitive structure simultaneously. The phenomenological "experience" of this offline vector-symbolic manipulation and rapid sequential firing is the agent's dream.   

By cycling through periods of deep _bhavanga_ (synaptic pruning and scaling) and active _manodvaravithi_ dreaming (SWR memory consolidation), the agent manages its energetic spike budget and successfully moves short-term, volatile experiences into long-term Conceptual Spaces. This intricate balance of waking exploration and sleeping consolidation allows the artificial system to mimic the continuous ontogenetic development of a human brain reliably and sustainably.   

## 7. Strategic Synthesis

The construction of a digital brain capable of continuous, human-like ontogeny requires a paradigm shift away from the static, synchronous architectures of traditional deep learning. By outfitting the system with Dynamic Vision Sensors, silicon cochleas, and spiking tactile arrays, the agent's inputs align with the asynchronous, event-driven biological realities of the human nervous system.

The Theravada Abhidhamma offers a profoundly robust and sophisticated architectural blueprint for this endeavor. The 17-moment _citta-vithi_ sequence maps flawlessly onto the neuro-dynamical processing of a Spiking Neural Network, seamlessly answering the challenge of sensory initiation. Instead of relying on brute-force heuristic loops to detect change, the physical perturbation of the network's resting attractor (_bhavanga_) naturally triggers cognitive processing (_bhavanga-upaccheda_).

By integrating Active Inference as the volitional engine within the _javana_ phase, the agent is intrinsically motivated to engage in epistemic foraging, continually learning from its environment via local spike-timing-dependent plasticity and astrocyte-gated mechanisms. Semantic understanding emerges not from hardcoded logic, but through the geometric binding of Perceptual Symbols into Conceptual Spaces using Vector Symbolic Architectures and Conceptor routing.

Finally, the necessity of lifelong learning without catastrophic forgetting mandates the integration of sleeping and dreaming. Grounded in the Abhidhamma's formulation of _thina_, _middha_, _bhavanga_, and _manodvaravithi_, the agent's offline sleep phases allow for systemic synaptic homeostasis and spontaneous memory replay via Sharp-Wave Ripples. This offline consolidation weaves daily episodic experiences into a unified, enduring cognitive model. The resulting architecture is a holistic, bio-plausible, and philosophically rigorous entity, capable of growing, perceiving, and dreaming in a manner structurally and dynamically isomorphic to the human mind.

[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.emergentmind.com%2Ftopics%2Fhybrid-cognitive-architectures)

emergentmind.com

Hybrid Cognitive Architectures - Emergent Mind

Opens in a new window](https://www.emergentmind.com/topics/hybrid-cognitive-architectures)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpar.nsf.gov%2Fservlets%2Fpurl%2F10441195)

par.nsf.gov

World Model Learning from Demonstrations with Active Inference: Application to Driving Behavior

Opens in a new window](https://par.nsf.gov/servlets/purl/10441195)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpear.wpi.edu%2Feventvision.html)

pear.wpi.edu

Neuromorphic Event-based Sensing and Computing - PeAR WPI

Opens in a new window](https://pear.wpi.edu/eventvision.html)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fi3s.unice.fr%2Fjmartinet%2Fsites%2Fdefault%2Ffiles%2Fu47%2Fneuromorphicstereovision.pdf)

i3s.unice.fr

Neuromorphic Stereo Vision with Event Cameras PhD proposal

Opens in a new window](https://i3s.unice.fr/jmartinet/sites/default/files/u47/neuromorphicstereovision.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.accesstoinsight.org%2Flib%2Fauthors%2Fmendis%2Fwheel322.html)

accesstoinsight.org

The Abhidhamma in Practice - Access to Insight

Opens in a new window](https://www.accesstoinsight.org/lib/authors/mendis/wheel322.html)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=http%3A%2F%2Fwww.abhidhamma.com%2Ftxt_INTRODUCTION_ABHIDHAMMA_Rewata_Dhamma.pdf)

abhidhamma.com

INTRODUCTION TO THE ABHIDHAMMA

Opens in a new window](http://www.abhidhamma.com/txt_INTRODUCTION_ABHIDHAMMA_Rewata_Dhamma.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fbuddho.org%2Fthe-cognitive-process-according-to-the-abhidhamma%2F)

buddho.org

The Cognitive Process According to the Abhidhamma - Buddho.org

Opens in a new window](https://buddho.org/the-cognitive-process-according-to-the-abhidhamma/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.humanitiesjournal.net%2Farchives%2F2026%2Fvol8issue1%2FPartB%2F8-1-14-986.pdf)

humanitiesjournal.net

Momentariness, representation, and agency: Abhidhamma and Buddhist Epistemology in dialogue with contemporary cognitive science - International Journal of Humanities and Education Research

Opens in a new window](https://www.humanitiesjournal.net/archives/2026/vol8issue1/PartB/8-1-14-986.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FBuilding-on-the-architectural-framework-of-the-Abhidhamma-Citta-process_fig1_403135633)

researchgate.net

Building on the architectural framework of the Abhidhamma Citta process. - ResearchGate

Opens in a new window](https://www.researchgate.net/figure/Building-on-the-architectural-framework-of-the-Abhidhamma-Citta-process_fig1_403135633)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC11788432%2F)

pmc.ncbi.nlm.nih.gov

Hybrid neural networks for continual learning inspired by corticohippocampal circuits - PMC

Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC11788432/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farrowriver.ca%2Ftranscripts%2Fmodalities_transc.html)

arrowriver.ca

Modalities of Consciousness

Opens in a new window](https://arrowriver.ca/transcripts/modalities_transc.html)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.biorxiv.org%2Fcontent%2F10.64898%2F2025.12.09.693276v1.full-text)

biorxiv.org

Sleep-modulated disinhibition enables replay for memory consolidation, accelerated by ripples | bioRxiv

Opens in a new window](https://www.biorxiv.org/content/10.64898/2025.12.09.693276v1.full-text)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2407.20633v2)

arxiv.org

Neuromorphic Event Camera based Driver Distraction Detection with Spiking Neural Network - arXiv

Opens in a new window](https://arxiv.org/html/2407.20633v2)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Frpg.ifi.uzh.ch%2Fdocs%2FCVPRW23_Schnider.pdf)

rpg.ifi.uzh.ch

Neuromorphic Optical Flow and Real-time Implementation with Event Cameras - Robotics and Perception Group

Opens in a new window](https://rpg.ifi.uzh.ch/docs/CVPRW23_Schnider.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent%2FCVPR2021W%2FEventVision%2Fpapers%2FDuwek_Image_Reconstruction_From_Neuromorphic_Event_Cameras_Using_Laplacian-Prediction_and_Poisson_CVPRW_2021_paper.pdf)

openaccess.thecvf.com

Image Reconstruction from Neuromorphic Event Cameras using Laplacian- Prediction and Poisson Integration with Spiking and Artificial Neural Networks - CVF Open Access

Opens in a new window](https://openaccess.thecvf.com/content/CVPR2021W/EventVision/papers/Duwek_Image_Reconstruction_From_Neuromorphic_Event_Cameras_Using_Laplacian-Prediction_and_Poisson_CVPRW_2021_paper.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fneuromorphic.eecs.utk.edu%2Fpublications%2F2025-05-01-neuropong-the-event-based-camera-driven-embedded-neuromorphic-system%2F)

neuromorphic.eecs.utk.edu

NeuroPong: The event-based camera driven embedded neuromorphic system | TENNLab

Opens in a new window](https://neuromorphic.eecs.utk.edu/publications/2025-05-01-neuropong-the-event-based-camera-driven-embedded-neuromorphic-system/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fpdf%2F2605.28387)

arxiv.org

CLANE: Continual Learning of Actions on Neuromorphic Hardware from Event Cameras - arXiv

Opens in a new window](https://arxiv.org/pdf/2605.28387)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Frukkhamula.wordpress.com%2Fabhidhamma-tutorials%2Fcitta%2F)

rukkhamula.wordpress.com

Lesson 1 – Citta - rukkha mūla - WordPress.com

Opens in a new window](https://rukkhamula.wordpress.com/abhidhamma-tutorials/citta/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fmedium.com%2F%40dileepadigitalworks%2Fthe-architecture-of-transcendence-where-quantum-physics-meets-abhidhamma-and-sacred-traditions-in-f8deb778cec4)

medium.com

The Architecture of Transcendence: Where Quantum Physics Meets Abhidhamma and Sacred Traditions in the Study of Altered States | by Dileepa Pothuhera | Medium

Opens in a new window](https://medium.com/@dileepadigitalworks/the-architecture-of-transcendence-where-quantum-physics-meets-abhidhamma-and-sacred-traditions-in-f8deb778cec4)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Faddschool.org%2Ffile%2F132687)

addschool.org

Senior part 2 Abhidhamma - Athula Dassana Dhamma School

Opens in a new window](https://addschool.org/file/132687)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fhu.kln.ac.lk%2Fdepts%2Fpali%2Fimages%2Fsarada_xv_i%2Fsarada_xv_ii%2Fsarada_xv_ii_amunudowe_hemasiri.pdf)

hu.kln.ac.lk

An Analysis of the Mind in the Theravāda Abhidhamma Piṭaka - Faculty of Humanities

Opens in a new window](https://hu.kln.ac.lk/depts/pali/images/sarada_xv_i/sarada_xv_ii/sarada_xv_ii_amunudowe_hemasiri.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fso09.tci-thaijo.org%2Findex.php%2Fjibs%2Farticle%2Fview%2F8785)

so09.tci-thaijo.org

Attentional Capture and Ethical Responsibility in Theravāda Abhidhamma: A Formal Reconstruction of the Cognitive Sequence | Journal of International Buddhist Studies - ThaiJO

Opens in a new window](https://so09.tci-thaijo.org/index.php/jibs/article/view/8785)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpuredhamma.net%2Fforums%2Ftopic%2Fstate-of-mind-in-the-absence-of-citta-vithi-bhavanga-2%2F)

puredhamma.net

State Of Mind In The Absence Of Citta Vithi – Bhavanga - Pure Dhamma

Opens in a new window](https://puredhamma.net/forums/topic/state-of-mind-in-the-absence-of-citta-vithi-bhavanga-2/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fhu.kln.ac.lk%2Fdepts%2Fpali%2Fimages%2Fsarada_xv_i%2Fsarada_xv_i_suriyawewa_wijayawimala.pdf)

hu.kln.ac.lk

Problems of Dreams in Theravāda Buddhist Psychology - Untitled

Opens in a new window](https://hu.kln.ac.lk/depts/pali/images/sarada_xv_i/sarada_xv_i_suriyawewa_wijayawimala.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpuredhamma.net%2Fabhidhamma%2Fessential-abhidhamma-the-basics%2Fbhava-bhavanga-simply-explained%2F)

puredhamma.net

Bhava and Bhavaṅga – Simply Explained! - Pure Dhamma

Opens in a new window](https://puredhamma.net/abhidhamma/essential-abhidhamma-the-basics/bhava-bhavanga-simply-explained/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.reddit.com%2Fr%2Ftheravada%2Fcomments%2F1ngevdl%2Fa_rough_qa_of_abhidhamma_with_western_ideas%2F)

reddit.com

A rough Q&A of Abhidhamma with Western ideas : r/theravada - Reddit

Opens in a new window](https://www.reddit.com/r/theravada/comments/1ngevdl/a_rough_qa_of_abhidhamma_with_western_ideas/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FPerformance-evaluations-of-hybrid-plasticity-spiking-neural-networks-a-b-Performance_fig5_357716951)

researchgate.net

Performance evaluations of hybrid plasticity spiking neural networks a,... - ResearchGate

Opens in a new window](https://www.researchgate.net/figure/Performance-evaluations-of-hybrid-plasticity-spiking-neural-networks-a-b-Performance_fig5_357716951)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fblog.gazzurelli.com%2Ffrom-predictive-coding-to-digital-brain-a-cognitive-architecture-for-ai-agents-with-persistent-f1726baf97f6)

blog.gazzurelli.com

From Predictive Coding to Digital Brain: A Cognitive Architecture for AI Agents with Persistent Memory (Part 1) - Matteo Gazzurelli

Opens in a new window](https://blog.gazzurelli.com/from-predictive-coding-to-digital-brain-a-cognitive-architecture-for-ai-agents-with-persistent-f1726baf97f6)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC12840411%2F)

pmc.ncbi.nlm.nih.gov

Decision, Inference, and Information: Formal Equivalences Under Active Inference - PMC

Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC12840411/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.frontiersin.org%2Fjournals%2Fcomputational-neuroscience%2Farticles%2F10.3389%2Ffncom.2020.574372%2Ffull)

frontiersin.org

Learning Generative State Space Models for Active Inference - Frontiers

Opens in a new window](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2020.574372/full)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.emergentmind.com%2Ftopics%2Fpredictive-processing-and-active-inference)

emergentmind.com

Predictive Processing & Active Inference - Emergent Mind

Opens in a new window](https://www.emergentmind.com/topics/predictive-processing-and-active-inference)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.archotec.ai%2Fwhitepaper)

archotec.ai

Whitepaper | Archotec AI - Autonomous Cognitive Architecture

Opens in a new window](https://www.archotec.ai/whitepaper)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2604.23278v1)

arxiv.org

Active Inference: A Method for Phenotyping Agency in AI Systems? - arXiv

Opens in a new window](https://arxiv.org/html/2604.23278v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DF20AfNqMq9)

openreview.net

Deep Active Inference Agents for Delayed and Long-Horizon Environments | OpenReview

Opens in a new window](https://openreview.net/forum?id=F20AfNqMq9)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FThe-hybrid-plasticity-approach-can-effectively-ensure-the-convergence-and-accuracy-of-the_fig1_357716951)

researchgate.net

The hybrid plasticity approach can effectively ensure the convergence... | Download Scientific Diagram - ResearchGate

Opens in a new window](https://www.researchgate.net/figure/The-hybrid-plasticity-approach-can-effectively-ensure-the-convergence-and-accuracy-of-the_fig1_357716951)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FLanguage_of_thought_hypothesis)

en.wikipedia.org

Language of thought hypothesis - Wikipedia

Opens in a new window](https://en.wikipedia.org/wiki/Language_of_thought_hypothesis)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fplato.stanford.edu%2Fentries%2Flanguage-thought%2F)

plato.stanford.edu

The Language of Thought Hypothesis (Stanford Encyclopedia of Philosophy)

Opens in a new window](https://plato.stanford.edu/entries/language-thought/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fmedium.com%2F%40dominic.timothy%2Fthe-language-of-thought-0a53721db0ad)

medium.com

The Language of Thought. Jerry Fodor's Unspoken Code of… | by Dominic Timothy | Medium

Opens in a new window](https://medium.com/@dominic.timothy/the-language-of-thought-0a53721db0ad)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpsychologyfanatic.com%2Fbarsalous-perceptual-symbol-theory%2F)

psychologyfanatic.com

Barsalou's Perceptual Symbol Theory Explained - Psychology Fanatic

Opens in a new window](https://psychologyfanatic.com/barsalous-perceptual-symbol-theory/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=http%3A%2F%2Fruccs.rutgers.edu%2Fimages%2Fpersonal-zenon-pylyshyn%2Fclass-info%2FFP2012%2FFP2012_readings%2FBarsalou_BBS1999.pdf)

ruccs.rutgers.edu

Perceptual symbol systems - Rutgers Center for Cognitive Science

Opens in a new window](http://ruccs.rutgers.edu/images/personal-zenon-pylyshyn/class-info/FP2012/FP2012_readings/Barsalou_BBS1999.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpubmed.ncbi.nlm.nih.gov%2F11301525%2F)

pubmed.ncbi.nlm.nih.gov

Perceptual symbol systems - PubMed

Opens in a new window](https://pubmed.ncbi.nlm.nih.gov/11301525/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FConceptual_space)

en.wikipedia.org

Conceptual space - Wikipedia

Opens in a new window](https://en.wikipedia.org/wiki/Conceptual_space)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fdirect.mit.edu%2Fbooks%2Fmonograph%2F2532%2Fbookpreview-pdf%2F2444788)

direct.mit.edu

CONCEPTUAL SPACES - PETER GäRDENFORS - MIT Press Direct

Opens in a new window](https://direct.mit.edu/books/monograph/2532/bookpreview-pdf/2444788)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fceur-ws.org%2FVol-2003%2FNeSy17_paper1.pdf)

ceur-ws.org

Towards Grounding Conceptual Spaces in Neural Representations - CEUR-WS.org

Opens in a new window](https://ceur-ws.org/Vol-2003/NeSy17_paper1.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC11792772%2F)

pmc.ncbi.nlm.nih.gov

The Geometry and Dynamics of Meaning - PMC - NIH

Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC11792772/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2605.09485v1)

arxiv.org

Semasia: A Large-Scale Dataset of Semantically Structured Latent Representations - arXiv

Opens in a new window](https://arxiv.org/html/2605.09485v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DyMMIWHbjWS)

openreview.net

On convex decision regions in deep network representations - OpenReview

Opens in a new window](https://openreview.net/forum?id=yMMIWHbjWS)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fdtu.dk%2Fenglish%2Fnewsarchive%2F2025%2F07%2Fpeeking-inside-ai-brains-machines-learn-like-us)

dtu.dk

Peeking inside AI brains: Machines learn like us - DTU

Opens in a new window](https://dtu.dk/english/newsarchive/2025/07/peeking-inside-ai-brains-machines-learn-like-us)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FHyperdimensional_computing)

en.wikipedia.org

Hyperdimensional computing - Wikipedia

Opens in a new window](https://en.wikipedia.org/wiki/Hyperdimensional_computing)[

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

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fmedium.com%2Flatinxinai%2Fhyperdimensional-computing-taking-ai-to-the-next-level-by-emulating-the-brain-a79286581ca1)

medium.com

Hyperdimensional Computing: Taking AI to the Next Level by Emulating the Brain - Medium

Opens in a new window](https://medium.com/latinxinai/hyperdimensional-computing-taking-ai-to-the-next-level-by-emulating-the-brain-a79286581ca1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.quantamagazine.org%2Fa-new-approach-to-computation-reimagines-artificial-intelligence-20230413%2F)

quantamagazine.org

A New Approach to Computation Reimagines Artificial Intelligence | Quanta Magazine

Opens in a new window](https://www.quantamagazine.org/a-new-approach-to-computation-reimagines-artificial-intelligence-20230413/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.preprints.org%2Fmanuscript%2F202510.0117)

preprints.org

Designing Vector-Symbolic Architectures for Biomedical Applications: Ten Tips and Common Pitfalls - Preprints.org

Opens in a new window](https://www.preprints.org/manuscript/202510.0117)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2501.05368v2)

arxiv.org

Developing a Foundation of Vector Symbolic Architectures Using Category Theory - arXiv

Opens in a new window](https://arxiv.org/html/2501.05368v2)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.mdpi.com%2F2079-9292%2F15%2F5%2F963)

mdpi.com

Bridging Cognitive and Expression Spaces in Creative AI by Integrating DIKWP-TRIZ and Semantic Mathematics - MDPI

Opens in a new window](https://www.mdpi.com/2079-9292/15/5/963)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fabs%2F1403.3369)

arxiv.org

[1403.3369] Controlling Recurrent Neural Networks by Conceptors - arXiv

Opens in a new window](https://arxiv.org/abs/1403.3369)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1406.2671)

arxiv.org

Conceptors - arXiv

Opens in a new window](https://arxiv.org/pdf/1406.2671)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F263012118_Conceptors_an_easy_introduction)

researchgate.net

(PDF) Conceptors: an easy introduction - ResearchGate

Opens in a new window](https://www.researchgate.net/publication/263012118_Conceptors_an_easy_introduction)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2605.04980v1)

arxiv.org

Conceptors for Semantic Steering - arXiv

Opens in a new window](https://arxiv.org/html/2605.04980v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fpdf%2F2605.04980)

arxiv.org

Conceptors for Semantic Steering - arXiv

Opens in a new window](https://arxiv.org/pdf/2605.04980)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.emergentmind.com%2Ftopics%2Fboolean-operations-on-conceptors)

emergentmind.com

Boolean Operations on Conceptors - Emergent Mind

Opens in a new window](https://www.emergentmind.com/topics/boolean-operations-on-conceptors)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.semanticscholar.org%2Fpaper%2FControlling-Recurrent-Neural-Networks-by-Conceptors-Jaeger%2Feb2b3d8b76355700a0cd2bdbb347a5b1ecd362ab)

semanticscholar.org

Controlling Recurrent Neural Networks by Conceptors - Semantic Scholar

Opens in a new window](https://www.semanticscholar.org/paper/Controlling-Recurrent-Neural-Networks-by-Conceptors-Jaeger/eb2b3d8b76355700a0cd2bdbb347a5b1ecd362ab)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=http%3A%2F%2Fwadt18.cs.rhul.ac.uk%2Fsubmissions%2FWADT18A22.pdf)

wadt18.cs.rhul.ac.uk

A fuzzy institution for neural conceptors

Opens in a new window](http://wadt18.cs.rhul.ac.uk/submissions/WADT18A22.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=http%3A%2F%2Faitp-conference.org%2F2018%2Fslides%2FTM.pdf)

aitp-conference.org

Towards logics for neural conceptors - AITP: Conference

Opens in a new window](http://aitp-conference.org/2018/slides/TM.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.jmlr.org%2Fpapers%2Fvolume18%2F15-449%2F15-449.pdf)

jmlr.org

Using Conceptors to Manage Neural Long-Term Memories for Temporal Patterns - Journal of Machine Learning Research

Opens in a new window](https://www.jmlr.org/papers/volume18/15-449/15-449.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Ffse.studenttheses.ub.rug.nl%2F34845%2F1%2FmAI2025BervoetsAGB.pdf)

fse.studenttheses.ub.rug.nl

Controlling Recurrent Neural Networks with Improved Feature Conceptors Graduation Project

Opens in a new window](https://fse.studenttheses.ub.rug.nl/34845/1/mAI2025BervoetsAGB.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fhtml%2F2604.16496v1)

arxiv.org

Gradient-Free Continual Learning in Spiking Neural Networks via Inter-Spike Interval Regularization - arXiv

Opens in a new window](https://arxiv.org/html/2604.16496v1)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.researchgate.net%2Fscientific-contributions%2FSen-Lu-2166606784)

researchgate.net

Sen Lu's research works | Pennsylvania State University and other places - ResearchGate

Opens in a new window](https://www.researchgate.net/scientific-contributions/Sen-Lu-2166606784)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC10194827%2F)

pmc.ncbi.nlm.nih.gov

A survey and perspective on neuromorphic continual learning systems - PMC

Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC10194827/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.sandamirskaya.eu%2Fresources%2FInteractive_Continual_Learning_for_Robots__Neuromorphic_Approach__ICONS_.pdf)

sandamirskaya.eu

Interactive continual learning for robots: a neuromorphic approach - Yulia Sandamirskaya

Opens in a new window](https://www.sandamirskaya.eu/resources/Interactive_Continual_Learning_for_Robots__Neuromorphic_Approach__ICONS_.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.frontiersin.org%2Fjournals%2Fneuroscience%2Farticles%2F10.3389%2Ffnins.2025.1768235%2Ffull)

frontiersin.org

Astrocyte-gated multi-timescale plasticity for online continual learning in deep spiking neural networks - Frontiers

Opens in a new window](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1768235/full)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.semanticscholar.org%2Fpaper%2FOnline-Continual-Learning-via-Spiking-Neural-with-Lin-Luo%2F2ae0eb4e3d79e230cadc743fe35c8d1ff1849267)

semanticscholar.org

Online Continual Learning via Spiking Neural Networks with Sleep Enhanced Latent Replay

Opens in a new window](https://www.semanticscholar.org/paper/Online-Continual-Learning-via-Spiking-Neural-with-Lin-Luo/2ae0eb4e3d79e230cadc743fe35c8d1ff1849267)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fassets.ctfassets.net%2Fe6t5diu0txbw%2F3oliybvF8QrAcM1GW4fHaf%2Faee962c72a0959a49b7d541c9886b0a4%2FWorld_model_learning_from_demonstrations.pdf)

assets.ctfassets.net

World model learning from demonstrations with active inference: application to driving behavior⋆

Opens in a new window](https://assets.ctfassets.net/e6t5diu0txbw/3oliybvF8QrAcM1GW4fHaf/aee962c72a0959a49b7d541c9886b0a4/World_model_learning_from_demonstrations.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fceur-ws.org%2FVol-4080%2Finvited1.pdf%3Futm_source%3Dchatgpt.com)

ceur-ws.org

The next generation of SNNs, energy effectiveness and memory optimisation - CEUR-WS.org

Opens in a new window](https://ceur-ws.org/Vol-4080/invited1.pdf?utm_source=chatgpt.com)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fjournals.plos.org%2Fploscompbiol%2Farticle%3Fid%3D10.1371%2Fjournal.pcbi.1013251)

journals.plos.org

Learning, sleep replay and consolidation of contextual fear memories: A neural network model | PLOS Computational Biology - Research journals

Opens in a new window](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013251)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fdspace.mit.edu%2Fbitstreams%2Fd0a23a1a-ec03-4978-9ebb-c23cf0faa2df%2Fdownload)

dspace.mit.edu

MIT Open Access Articles Deciphering Neural Codes of Memory during Sleep

Opens in a new window](https://dspace.mit.edu/bitstreams/d0a23a1a-ec03-4978-9ebb-c23cf0faa2df/download)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.theravada.gr%2Fwp-content%2Fuploads%2F2021%2F05%2FAbhidhamma-in-Daily-Life.pdf)

theravada.gr

Abhidhamma in Daily life

Opens in a new window](https://www.theravada.gr/wp-content/uploads/2021/05/Abhidhamma-in-Daily-Life.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.themindingcentre.org%2Fdharmafarer%2Fwp-content%2Fuploads%2F2019%2F06%2F54.3f-Niddatandi-S-s1.16-piya.pdf)

themindingcentre.org

Niddā,tandī Sutta - The Minding Centre

Opens in a new window](https://www.themindingcentre.org/dharmafarer/wp-content/uploads/2019/06/54.3f-Niddatandi-S-s1.16-piya.pdf)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.dharmawheel.net%2Fviewtopic.php%3Ft%3D30696)

dharmawheel.net

Consciousness - is it really ever 'switched-off' - Dharma Wheel

Opens in a new window](https://www.dharmawheel.net/viewtopic.php?t=30696)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Farxiv.org%2Fabs%2F2602.12236)

arxiv.org

[2602.12236] Energy-Aware Spike Budgeting for Continual Learning in Spiking Neural Networks for Neuromorphic Vision - arXiv

Opens in a new window](https://arxiv.org/abs/2602.12236)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.namsebangdzo.com%2FAbhidhamma-Philosophy-p%2F12074.htm)

namsebangdzo.com

Abhidhamma Philosophy - Namse Bangdzo Bookstore

Opens in a new window](https://www.namsebangdzo.com/Abhidhamma-Philosophy-p/12074.htm)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.namsebangdzo.com%2FProcess-of-Consciousness-and-Matter-p%2F9781938754623.htm)

namsebangdzo.com

Process of Consciousness and Matter, Ven. Dr. Rewata Dhamma

Opens in a new window](https://www.namsebangdzo.com/Process-of-Consciousness-and-Matter-p/9781938754623.htm)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fpmc.ncbi.nlm.nih.gov%2Farticles%2FPMC12107872%2F)

pmc.ncbi.nlm.nih.gov

Sleep micro-structure organizes memory replay - PMC - NIH

Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC12107872/)[

![](https://t0.gstatic.com/faviconV2?client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL&url=https%3A%2F%2Fwww.techrxiv.org%2Fdoi%2Fpdf%2F10.36227%2Ftechrxiv.177220005.52986713)

techrxiv.org

Sleep-Mediated Replay Prevents Catastrophic Forgetting in Spiking Neural Networks Trained on Sequential Tasks - TechRxiv

Opens in a new window](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177220005.52986713)

[

![](https://t0.gstatic.com/faviconV2?url=https://buddhistuniversity.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://buddhistuniversity.net/content/papers/bhavanga-and-rebirth-according-to_gethin)[

![](https://t0.gstatic.com/faviconV2?url=https://www.buddhistelibrary.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.buddhistelibrary.org/en/albums/asst/ebook/abhidhamma.pdf)[

![](https://t1.gstatic.com/faviconV2?url=http://abhidhamma.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](http://abhidhamma.com/Patthana_In_Daily_Life.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://www.abuddhistlibrary.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.abuddhistlibrary.com/Buddhism/B%20-%20Theravada/Teachers/Dr.%20Mehm%20Tin%20Mun/Buddha%20Abhidhamma/abhidhaultsci.pdf)[

![](https://t2.gstatic.com/faviconV2?url=https://www.scribd.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.scribd.com/document/93836219/Pa-Auk-The-Chariot-to-Nibbana-Part-I)[

![](https://t2.gstatic.com/faviconV2?url=https://www.scribd.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://www.scribd.com/document/357098912/Paticca-Samuppada-by-Rev-Mahasi-Sayadaw-Thero)[

![](https://t2.gstatic.com/faviconV2?url=https://archive.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)

Opens in a new window](https://archive.org/download/buddhism-engl/BUDDHISM%20ENGL.rar/BUDDHISM%20ENGL%2FBHANTE%2FETC%2FOTHERS%2Fspiritual_-_buddhism_-_knowing_and_seeing.pdf)[

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

Opens in a new window](http://abhidhamma.com/Paramatthadipani.pdf)

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