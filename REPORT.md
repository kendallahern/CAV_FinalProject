# Formal Verification of Neural Network Robustness Using SMT-Based Methods

Kendall Ahern - December 2025

Computer Aided Verification Final Project

**GitHub Repo Link: https://github.com/kendallahern/CAV_FinalProject**

---

## Abstract

For my final project, I wanted to investigate the problem of formally verifying the local adversarial robustness of neural networks using SMT solvers. Motivated by the limitations of purely empirical adversarial attack methods, I designed and implemented a verification funnel that begins with fast, approximate screening before narrowing down to exact constraint-solving–based guarantees. The approach combines projected gradient descent (PGD) attacks, interval bound propagation (IBP), and formal SMT reasoning in Z3 to assess the robustness of models at selected input points. I learned about these attack methods at work, and not only wanted to learn more about them, but wanted to see how I could apply some of the concepts to what we learned in this class. I applied this methodology to two types of networks: a simple fully connected model (TinyNet) and a convolutional MNIST classifier (MiniNet). Through these experiments, I observed both the strengths and limitations of Z3 as a verification backend, particularly in relation to networks with a large number of ReLU activations and convolutional structure. Although I initially planned to implement a scalable verification pipeline using Marabou, Docker, and Kubernetes, I ultimately focused my efforts on understanding and characterizing the behavior of Z3 in exact local robustness verification settings. The following report presents my methodology, results, insights, and proposed future directions toward a more scalable and specialized verification system.

## Introduction

As neural networks become increasingly integrated into high-stakes domains such as autonomous driving, medical diagnostics, and security systems, ensuring their reliability has become an essential research focus. A particularly concerning vulnerability arises from adversarial perturbations. These are small, carefully crafted changes to an input that can cause a model to produce an incorrect output despite the perturbation being imperceptible to the human eye. Although empirical defenses and attack strategies are well studied, empirical approaches alone cannot provide absolute guarantees. To produce meaningful assurances, one must turn to methods that can formally prove whether or not a model is robust around a particular input.

The goal of my project was to explore formal guarantees of local adversarial robustness. Local robustness concerns whether a network’s prediction remains unchanged within an $\epsilon$-bounded neighborhood of a specific input. more formally written, a model $f$ is considred locally robust at a point $x_0$ if $\forall x (||x-x_0||_{\infty} \leq \epsilon) \rightarrow f(x) = f(x_0)$.

If this implication holds, the model is provably safe against adversarial perturbations of size up to $\epsilon$. However, if it does not hold, the corresponding counterexample provides concrete evidence of vulnerability. Since modern neural networks often contain thousands of piecewise-linear ReLU activations that induce an exponential branching structure (as introduced to me in the literature), the primary challenge in this type of approach is that checking this property requires exploring a highly non-liner decision space. To address these challenges, I developed and tested a multi-stage verification funnel that uses fast screening methods to reduce the number of candidate points requiring formal verification. This approach allowed me to study when Z3 is effective, when it fails, and what architectural features most strongly influence solver performance. Although my original vision involved a large-scale distributed verification system built around Marabou, my experiments uncovered significant complexities that diverted my attention toward understanding Z3’s capabilities and limitations. This paper provides a comprehensive narrative of my methodology, results, and lessons learned.

## Background

### Adversarial Robustness and Local Verification

Adversarial robustness has become a central topic in modern machine learning research, especially because high-capacity models tend to be brittle in counterintuitive ways. While global robustness is extremely difficult to guarantee for real-world neural architectures, local robustness offers a tractable alternative by focusing on specific inputs rather than entire distributions. For a fixed input $x_0$ the question becomes whether any point in the $\epsilon$-ball around $x_0$ induces a misclassification. If no such point exists, the model is provably robust on that example.

Local robustness verification is appealing because it transforms a large-scale learning problem into a fixed-input analysis problem. However, the core difficulty still remains: neural networks with ReLU activations create a piecewise-linear computation graph with a combinatorial explosion of linear regions. Each ReLU can be either active or inactive depending on the input, and in the worst case a model with $n$ ReLUs may generate up to $2^n$ such regions. SMT solvers attempt to reason over these cases symbolically, which is why specialized verification tools and heuristics are crucial for handling non-trivial networks.

### Empirical Attacks as a Screening Mechanism

Although empirical methods cannot provide formal guarantees, they are extremely valuable for pre-filtering cases that are obviously non-robust. In this project, I used projected gradient descent (PGD) as a first-stage testing method. PGD attempts to find an adversarial perturbation by iteratively updating the input in the direction of the gradient of the loss. The update rule is given by

$x_{t+1}=\Pi _{B_{\epsilon}(x_0)} (x_t + \alpha \cdot \text{sign} (\nabla _x L(f(x_t),y)))$

where $\Pi _{B_{\epsilon}(x_0)}$ is the projection onto the $\epsilon$-ball. If PGD succeeds in finding an adversarial example, then formal verification is unnecessary for that input. On the other hand, if PGD fails, the input becomes a candidate for deeper analysis.

### Interval Bound Propagation as an Intermediate Filter

Following PGD, I applied interval bound propagation (IBP) to obtain symbolic upper and lower bounds on neuron activations. IBP works by tracking an interval $[l,u]$ for each tensor through every layer of the network. Linear layers propagate intervals exactly, while ReLU layers clamp the lower bounds to zero. IBP does not attempt to reason about correlations between neurons which makes it a conservative but computationally cheap over-approximation.

IBP is valuable because it can quickly determine whether all ReLUs are stable within the perturbation region. A ReLU is stable if its pre-activation interval is entirely above zero or entirely below zero. Stable ReLUs do not contribute branching complexity during SMT solving. When many ReLUs are unstable, the SMT solver faces a difficult combinatorial search problem. Thus, IBP acts as a predictor of verification difficulty and helps avoid wasting time on hopeless cases.

### SMT-Based Verification in Z3

Z3 is a general-purpose SMT solver capable of reasoning about real arithmetic and logical constraints - we spent several homework assignments working with Z3 and I wanted to use something familiar in my project and I did not take into account some of its downsides (more on this at the end of the paper). A neural network can be expressed in terms of affine transformations and piecewise linear operations. An affine transformation is a geometric transformation that preserves lines and parallelism, but not necessarily distances or angles. For each layer, I created constraints that encoded both the linear relationships and the ReLU activation patterns. A ReLU is modeled using conditional constraints of the form $x= \text{max} (0,z)$,

which is implemented by enforcing

$x\geq 0$, $x\geq z$, $x=0$ or $x=z$.

These disjunctions create branching in the solver. Exact verification requires Z3 to explore or prune these branches optimally. Compared to more specialized neural network verifiers, Z3 lacks built-in heuristics for neural structure, but its expressiveness makes it a flexible tool for experimentation. My hope was that with a small enough network, such as the MiniNet and TinyNet ones I created, Z3 would be able to analyze the robustness. 

### Verification Funnel and System Architecture

To manage the difficulty of exact verification, I constructed a three-stage funnel that progressively narrows the number of inputs sent to Z3. The first stage consists of empirical adversarial attacks, which quickly identify points that are obviously non-robust. The second stage uses IBP to determine whether the network’s activation structure is sufficiently stable for feasible SMT solving. Only inputs that pass both stages proceed to the third stage, where exact SMT queries are constructed.

The rationale behind this architecture is that pure SMT solving is often computationally prohibitive when applied indiscriminately. By filtering inputs aggressively, I wanted to reduce solver load while obtaining meaningful robustness guarantees. This imitates practical verification systems that combine heuristic screening with formal reasoning in order to scale to nontrivial networks.

In implementing this funnel, I created a library capable of exporting PyTorch model parameters, encoding them symbolically in Z3, and systematically varying the perturbation radius $\epsilon$. The system also included detailed logging to track solver performance, runtime, and query outcomes. Although this pipeline was designed to be extensible, including planned support for ONNX export and integration with Marabou, the bulk of my experiments focused on Z3-based analysis.

To verify robustness, I encoded the negation of the robustness property in Z3 by asserting the existence of an input within the $\epsilon$-ball around $x_0$ that causes the network’s predicted class to differ from $f(x_0)$ allowing the solver to search for adversarial counterexamples.

## Methods, Results, and Discussion

### Case Study #1: Verification of TinyNet

TinyNet is a fully connected network consisting of several layers with ReLU activations. Its relatively small size and lack of convolutional structure made it a good candidate for detailed formal verification experiments. To perform verification, I exported its weights and biases from PyTorch and translated each layer into Z3 constraints. For every layer, I represented the affine transformations

$z^{(k+1)} = W^{(k)} x^{(k)} + b^{(k)}$

and then encoded the ReLU constraints required to obtain  $x^{(k+1)}$.

I systematically tested several inputs drawn from the MNIST dataset and varied $\epsilon$ to observe how solver performance changed. For example, At very small radii such as $\epsilon =0.001$, I found that the model was robust for many inputs. PGD did not find attacks, IBP showed that most ReLUs were stable, and Z3 quickly returned UNSAT, indicating that the input was provably robust. However, at moderate radii such as $\epsilon =0.01$, the situation became a bit more complicated. Often times, PGD found adversarial examples immediately, reflecting the simplicity of the network. In cases where PGD did not find an attack, IBP frequently identified large numbers of unstable ReLUs, and Z3 typically returned SAT with valid adversarial counterexamples. These cases confirmed that the model had very limited inherent robustness.

These results are consistent with the expected behavior of small ReLU networks, since tiny perturbation radii naturally preserve local linearity and keep most activations stable, whereas slightly larger radii expand the reachable region enough to cross decision boundaries, destabilize ReLU patterns, and expose adversarial examples that gradient-based and SMT-based methods can readily find.

TinyNet allowed me to characterize success patterns for Z3: the solver performed well when many ReLUs were stable but suffered when even a modest number became unstable. This suggested that for any meaningful $\epsilon$, fully connected models of moderate depth can quickly exceed Z3’s practical limits unless strong relaxation techniques are used.

### Case Study #2: Verification of a Convolutional MNIST Model

The second major model I examined was a convolutional neural network, MiniNet, trained on MNIST. I have worked with this model numerous times in other course, especially as a learning tool in Deep Learning, so I thought it would be interesting and also beneficial to try and relate it/use it to some of the work from this class. However, I knew that convolutional models would pose significantly greater challenges due to their larger parameter counts and more complex layer structure. To adapt the network for verification, I exported it to a flattened representation that Z3 could process, but this transformation introduced considerable overhead because each convolutional filter needed to be unrolled into explicit affine constraints.

When performing PGD-based screening, I observed that some inputs were robust to initial attacks, but many were not. The more interesting cases were those where PGD failed to find adversarial examples. For these points, like the case study before, I applied IBP. The resulting bounds were typically much looser than for TinyNet. Many convolutional activations had extremely broad intervals, and ReLUs were unstable even at very small perturbation radii. This meant that the SMT queries constructed for these points were inherently complex.

During exact verification, Z3 struggled even more visibly than in the TinyNet case. For $\epsilon = 0.01$, many queries exceeded the ten-minute timeout I had initially set. The solver frequently returned unknown or failed to converge due to branching complexity. Interestingly, when I set $\epsilon = 0.0$ and asked the solver to simply re-evaluate the model with fixed input, Z3 solved the query immediately, which confirmed the correctness of the encoding. This contrast emphasized the central difficulty: even tiny relaxation of the input domain dramatically increases the search space. At the time I am writing this report, I have extended the timeout for a single default run to 700 days of a timeout to see if I was able to get a convergence at all. Thus far, I am 18 hours in and haven't seen one, but I will have a script running on my somputer to keep my mouse active and see if I can reach a SAT or UNSAT decsion before I will be presenting my project.

Overall, the convolutional model confirmed a key insight that was suggested by the paper, but also what I initially suspected: Z3 is poorly suited for networks with significant depth or convolutional structure unless substantial pre-processing or abstraction is applied. The results suggested that a more specialized verifier would be required for any scalable analysis. At the end of this project, I tried to work with Marabou - more in the Future Directions section of this report.

### Experimental Observations and Interpretation

Across both models, I observed several consistent patterns regarding robustness and solver behavior. Empirical PGD attacks proved to be an extremely efficient first-stage filter. Whenever PGD identified an adversarial example, Z3 would inevitably return SAT, often in much longer time than PGD required. Conversely, when PGD failed, the input was not guaranteed to be robust, but it was worth sending forward to IBP and possibly to Z3.

IBP provided a middle layer of analysis that offered insight into the solver’s likely performance. When IBP indicated that nearly all ReLUs were stable, Z3 solved the query quickly. When many ReLUs were unstable, Z3 either returned SAT with a counterexample or exceeded the timeout. This relationship highlighted the importance of neuron stability for formal verification. It also illustrated why specialized tools often incorporate advanced bound tightening methods, such as dual optimization or mixed-integer linear programming relaxations.

The final stage, exact SMT solving in Z3, behaved predictably according to network size and layer type. TinyNet, being a small fully connected network, was verifiable for small regions. MiniNet, despite being moderately sized by modern standards, was significantly more difficult to verify. The solver frequently exceeded time limits or returned unknown. These results aligned with expectations from the literature indicating that general-purpose SMT solvers face substantial scalability challenges when applied to neural networks with many activations.

### Limitations of Z3

As a result of this experiment, I identified several fundamental limitations associated with using Z3 for neural network verification. The first major issue was the combinatorial explosion arising from ReLU case splits. Each unstable activation requires Z3 to consider separate branches corresponding to active and inactive states. In large networks, these branches accumulate into an astronomically large search tree that is impractical to explore fully.

A second limitation was Z3’s lack of native support for convolutional operations. I had to unroll convolutional layers manually into affine transformations, which dramatically increased the number of constraints and resulted in slower solving times. Specialized verifiers, by contrast, should handle convolutional layers more intelligently and apply solver-side optimizations that Z3 does not, as mentioned in some of the prior work discussed by the papers.

Third, I encountered numerical issues when Z3 attempted to reason about floating-point arithmetic. Small numerical discrepancies occasionally led to unknown outcomes or unstable solver behavior. Although it is possible to express network weights as rationals to avoid floating-point issues, this approach increases constraint size and slows down solving.

Finally, timeout behavior was a practical concern. If you look at some of the logs in file like `Code/week4/results/z3/case_924_eps_20_20251124_203633.json` - it is important that you look at the ones with the 1120-1124 date/time stamp because that is when I extended the time limit - you can see that even with 200,000 seconds (~2.5 days which is about the only leeway I had in running tests for a project with this deadline), Z3 still found the cases ambiguous. Many queries related to convolutional networks did not finish within ten minutes, making Z3 unsuitable for systematic verification of more than a handful of inputs. These empirical observations motivated my plan to migrate toward a more capable tool.

### Engineering Challenges and Debugging Process with Z3

To be honest, I spent a lot of time on this project debugging and iterating to improve the performance of Z3 as working wiht it introduced several practical challenges. One of the first issues I encountered was the sensitivity of the solver to how constraints were encoded. Even small modeling choices — such as whether to encode the property using straightforward output inequalities or by formulating the negation of the property instead - had a large effect on solver stability and runtime. In several cases, Z3 appeared to “stall,” not failing outright but also not making progress for minutes at a time, which made it difficult to distinguish between a genuinely hard instance and an encoding that was silently causing the solver to struggle.

Another challenge involved managing numerical precision. Because neural-network verification requires reasoning about real-valued inequalities, Z3’s handling of floating-point comparisons sometimes produced unexpected behavior, especially when logits were very close. To mitigate this, I added a small margin $\delta$ to enforce strict separations between classes. However, even with this modification, certain inputs produced borderline cases that required repeated adjustments to the encoding.

Debugging was also complicated by the difficulty of extracting meaningful intermediate information from Z3. Unlike a traditional program with breakpoints, solver execution with Z3 is opaque - meaning I could really only figure out a way to get information in the form of a solution at the end of the run - and when a query timed out, Z3 provided little insight into whether the issue was constraint explosion, unstable branching, or simply an especially ambiguous input. I introduced additional logging around the construction of each constraint and dumped the SMT formulas for problematic cases so that I could go back through and manually inspect them. This process helped to identify several mistakes - primarily inconsistent bounds and missing conjunctions - that were preventing the solver from reaching a definitive SAT or UNSAT result.

Finally, scaling up the analysis beyond individual test points surfaced more subtle workflow issues. Running batches of verification tasks revealed that certain images or robustness radii regularly triggered long solver runtimes, indicating weaknesses in either the encoding strategy or the network’s structure. These cases became valuable for iteratively improving the SMT generation code and for refining my understanding of how Z3 behaves on neural-network constraints. 

### More Details on System Implementation for Z3

In this section, I want to talk a little bit more about the engineering behind the building the verification system that powered all experimental results. The verifier was constructed as a multi-stage pipeline: PyTorch models were exported, parameters were extracted, affine layers were encoded symbolically, and ReLU constraints were wrapped in solver-friendly disjunctions. Each part required careful design to preserve correctness while keeping the constraint system tractable. I began by implementing a direct PyTorch-to-Z3 parameter extraction routine. PyTorch stores layer weights and biases as tensors, so my export function converted them into NumPy arrays and then into plain Python lists usable by Z3. Every weight matrix $W$ and bias vector $b$ was translated into a constraint of the form $z=Wx+b$. Because MiniNet included convolutional layers, I had to implement flattening and indexing logic that expanded convolutions into linear constraints by manually performing an im2col transformation. This produced the affine maps required for Z3 to treat all layers uniformly.

For ReLU layers, I encoded each activation $y= \text{max} (0,x)$ using the standard disjunctive linear formulation. Z3 does not natively support ReLU, so each node required a binary case split: one branch where $x \leq 0$ forces $y=0$, and the other branch where $x \geq 0$ forces $y=x$. This is ehere the combinarotila explosion origivnates. I also added the adversarial constraint $|x-x_0|_{\infty} \leq \epsilon$ by generating a pair of linear inequalities for each dimension. The overall structure of the encoding was a long chain of affine transformations with interleaved ReLU disjunctions. All of these were assembled into a Z3 solver object, with careful bookkeeping/tracking to ensure that each variable was indexed correctly, since flattening mistakes caused numerous subtle bugs throughout my development process and testing phase. 

The code is too long to copy and paste here, but I wanted to highlight the core pieces: parameter extraction code, the affine layer transformer, the ReLU disjunction builder, and the adversarial region encoder. Together they formed a complete symbolic representation of MiniNet suitable for SMT-based robustness checking.

### Attempted Marabou Integration

After noticing a lot of the short comings with Z3, I decided to attempt to integrate my design with Marabou - even though it was not a solver that we used in class. Unfortunatly, I ran into a lot of problem integrating Marabou and it consumed a substantial portion of the engineering effort at the end of the project. This was largely due to the inconsistencies between its Python API, ONNX parsing behavior, and underlying C++ structures. My workflow began with exporting MiniNet to ONNX, which Marabou often rejected due to unsupported operators or malformed shapes. Once the model loaded, MarabouNetwork’s flattening of variables produced mismatch errors between declared inputVars and outputVars that required patching the parser.

The most disruptive issue involved MarabouCore’s EquationType. The Python API expected the constructor to accept an integer enumeration, but the runtime bindings lacked the attribute entirely. Attempting to construct equations triggered an AttributeError, leading me into a long cycle of debugging. I attempted manual code edits inside InputQueryBuilder.py to adjust how EquationType was retrieved. After applying those patches, I repeatedly deleted pycache files, restarted the virtual environment, and searched the entire codebase with grep to confirm the presence of the updated lines. But Marabou continued to import stale bytecode or to retrieve symbols differently than expected. Some fixes introduced parentheses imbalances, triggering SyntaxError on import. Other edits silently failed because Marabou loaded a different version of the module from a nested directory inside the Docker image. Ultimately, the issue appeared to stem from a deeper mismatch in the C++ bindings that the Python wrapper could not correct manually. On the bright side I guess, this experience taught me more about how tight coupling between compiled and interpreted languages can make research codebases brittle and challenging to debug.

## Additional Information

### Notes on Convolutional Constraints and Unrolling

Much of the difficulty in verifying MiniNet came from its convolutional layers. Although CNNs are conceptually simple, their verification requires either special-purpose convolution-aware solvers or explicit unrolling into dense affine constraints. Because I used Z3 and Marabou, both of which treat convolutions as generic operations, the only viable approach was im2col unrolling. This transformation expands each convolution into a very large linear map, multiplying the number of variables and constraints by the filter size, spatial dimensions, and channel count. The result is that small kernel operations become enormous constraint blocks, dramatically loosening bounds and increasing solver branching. In abstract-interpretation terms, these broadened affine sets act like zonotopes—geometric shapes defined by a center and a collection of generator vectors—which tend to expand rapidly as they propagate through the network. Even IBP becomes coarse once convolutional layers expand the feasible region.

This helps explain why MiniNet’s verification difficulty increased sharply after the first few layers. The network’s convolutions turn a small input uncertainty region into a high-dimensional polytope with many facets. The more such layers are unrolled, the harder it becomes to maintain tight bounds on ReLU stability, and the more case splits Marabou or Z3 must consider. The corresponding constraint explosion is entirely predictable from the theory of linear+ReLU networks, but seeing it manifest in practice made the limitations of current tools extremely clear.

### Complexity Analysis of ReLU Networks

The theoretical difficulty of verifying ReLU networks - that was mentioned in numerous papers - played out directly in my experiments. ReLU networks define piecewise-linear functions whose number of linear regions grows exponentially with the number of ReLUs; in the worst case, $n$ ReLU unites induce up to $2^n$ activation patterns. Although real networks rarely reach this bound, MiniNet’s layered structure still produced rapidly increasing region counts, making robustness queries difficult. ReLU stability — which is whether a ReLU is guaranteed active or inactive throughout the adversarial region — was a major determinant of solver performance. When IBP predicted many unstable ReLUs, SMT solving time increased dramatically due to the need to branch on those activation patterns.

Here's a small example that illustrates the phenomenon: even a single layer with three ReLUs has eight possible activation patterns, each corresponding to a distinct set of linear constraints. A verifier must either explore them or prove that many are infeasible. As you can see, when scaled to hundreds of ReLUs, this creates the combinatorial blow-up that dominated most of my runtime results. When I was able to understand this behavior, it helped to contextualize the observed gap between $\epsilon = 0$ and $\epsilon > 0$ verification difficulty - explained a bit in the case study below. 

#### Case Study 3: $\epsilon=0$ vs $\epsilon > 0$ Verification

One of the most striking empirical findings was the sharp contrast between $\epsilon = 0$ verification and $\epsilon > 0$ verification. When $\epsilon = 0$, the input is fixed, eliminating all adversarial constraints. As a result, every ReLU input is a known scalar, making most ReLUs provably stable. Z3 solved these cases almost instantly because no branching was required. In contrast, even a very small positive $\epsilon$ introduces a hypercube of possible inputs, forcing the solver to propagate intervals through the network and re-evaluate ReLU stability. In many regions, even tiny input changes can flip activation states. This forces the solver to consider a large number of potential activation patterns, leading to dramatically higher runtime. The difference between these two regimes is pretty drastic and helps illustrate the inherent difficulty of robustness verification: exact reasoning about neighborhood of points is far harder than reasoning about fixed points.

## Conclusion

Through reviewing the different sources and some experimentation of my own, I was able to develop a clearer perspective on different verification techniques. Reluplex, the earliest solver, provided the some of conceptual foundation but it quickly became clear that is was not designed for modern network sizes. Marabou, Reluplex's direct successor, adds engineering improvements and better branching heuristics, but still remains sensitive to model structure, import format, and constraint cleanliness (I was unable to complete experiments with this, but this is what I understand from the papers). Z3, while expressive, is not specialized for neural networks and often returns “unknown” on problems involving large numbers of disjunctions. 

From the papers, there are other tools out there that I was unable to investigate or mess around with in this project. For example and in contrast to the tools above, tools like α-β-CROWN and its variants use optimization-based and abstract-interpretation-based methods that scale much better for global robustness but cannot provide exact SMT-level proofs in all cases. MILP-based (Mixed Integer Linear Programming) approaches offer completeness but scale poorly. Abstract interpretation methods like DeepPoly and interval-based methods provide scalable certificates but produce conservative bounds.

My project highlighted the tradeoff: Z3 and Marabou provide exact proofs but struggle with networks like MiniNet, whereas newer research tools scale to larger models but offer less precision or depend on sophisticated bound propagation techniques. This comparison motivated the use of a “verification funnel,” where cheaper methods filter trivial cases before invoking expensive solvers.

My three-stage verification funnel—PGD, IBP, and SMT—proved to be an effective architecture. Each method eliminated a different subset of inputs: PGD quickly found obvious adversarial examples; IBP proved robustness for many small ε cases; and SMT handled the hardest remaining queries. The empirical results showed a dramatic reduction in SMT load. By logging per-stage runtimes, I observed that PGD typically completed in milliseconds, IBP took tens of milliseconds, and SMT ranged from seconds to hundreds of seconds depending on ReLU instability. A clear pattern emerged: as $\epsilon$ increased, PGD found more attacks, IBP proved fewer points, and SMT runtimes increased significantly. Graphs of unstable ReLU counts aligned closely with solver runtime plots, reinforcing the theory that ReLU stability governs complexity.

A histogram of Z3 runtime showed a bimodal distribution: easy ε = 0 cases solved instantly, while even moderately perturbed inputs caused the solver to explore many activation branches. Overall, the funnel prevented a combinatorial explosion by reducing the number of SMT queries to a manageable subset.

Through this project, I developed an end-to-end verification pipeline that combines empirical, symbolic, and formal methods to assess the local robustness of neural networks. I applied this system to both fully connected and convolutional models to study how adversarial vulnerability and solver behavior vary across architectures. The results demonstrated that Z3 is capable of producing exact robustness proofs for small networks and small perturbation radii but struggles significantly with larger or more complex architectures. These findings underscore the need for specialized verification tools tailored to neural networks - which was something mentioned throughout the papers that I read and re-emphasized through my own experiments.

Although my original plan included deploying a large-scale verification framework using Marabou and distributed computation, the challenges encountered during Z3-based verification provided valuable insight into the fundamental difficulties of formal reasoning in high-dimensional, piecewise-linear systems. The limitations I observed, including branching explosion, convolutional complexity, and numerical instability, motivate what would be the next phase of my work: migrating to a more scalable verification backend and integrating it into a containerized and distributed system.

Overall, this project highlights both the promise and challenges of formal robustness verification. It demonstrates the feasibility of obtaining exact guarantees for certain models and inputs while also emphasizing the need for advanced tooling and computational infrastructure to scale these methods to modern deep learning architectures.

This project taught me a lot about the practical realities of formal verification. First, verification is fundamentally sensitive to model structure, and seemingly small architectural decisions (e.g., convolution placement or layer width) can drastically alter solver difficulty. Second, debugging verification tools requires comfort with multiple layers of abstraction—Python, ONNX, C++ bindings, solver logs, and symbolic semantics. Third, containerization is valuable but does not prevent dependency mismatches or stale bytecode issues. Fourth, incremental debugging is essential: verifying individual layers, variables, and constraints before assembling the full network saves enormous time. Finally, hybrid verification pipelines strike a balance between tractability and rigor, allowing SMT solvers to focus on the hardest cases.

For future students entering this space, I would emphasize the importance of patience, careful logging, and a deep understanding of both numerical and symbolic representations. Formal verification is incredibly rewarding but demands meticulous attention to detail, especially when working with evolving research tools. I spent a lot of debugging time spiralling and circling around the core issue, so I would strongly recommend that you have some expereince with the tools before diving into this project.

### Future Extensions Using Marabou, Docker, and Kubernetes

Early in the project, my "long-shot" goal was to build a large-scale verification system based on Marabou, a solver explicitly designed for neural network verification because it integrates advanced heuristics such as ReLU phase inference, symbolic bound tightening, and specialized branching strategies. Therefore, it is significantly more efficient than Z3 for networks with deep architectures or convolutional layers. My plan was to first export the trained networks to ONNX and then load them into Marabou’s Python interface. Once basic verification functionality was established, I planned to containerize the pipeline using Docker to ensure consistent environments, reproducibility, and easier deployment.

The final vision involved deploying a distributed verification service on a Kubernetes cluster. Each input point requiring verification would be assigned to an independent pod, allowing hundreds of queries to be processed concurrently. Failed or timed-out pods would automatically restart, and results would be logged through a centralized collection system. Such a design would transform the system into a scalable verification engine capable of handling realistic workloads. Although I made progress on the ONNX export and initial Marabou setup, issues related to model formatting and solver configuration prevented me from completing this part of the project within the available time.

Nevertheless, the architectural design remains a promising future direction. A distributed verification system built around Marabou would overcome many of the limitations encountered with Z3, especially regarding convolutional models. It would also allow researchers to examine robustness at scale, enabling statistical assessments of model safety across large datasets.

## Appendix: 

#### Additional Technical Background, Experiments, and Supplementary Notes

This appendix provides extended technical detail and supplemental analysis derived from the literature I reviewed and the experiments I conducted during the development of the verification system. While the main body of the report focuses on the evaluation results and system design, the appendix presents deeper theoretical context, mathematical elaborations, and experimental findings that support and extend the earlier discussion. These notes include expanded explanations of verification theory, solver behavior, convolutional unrolling effects, robustness notions, and the empirical characteristics of MiniNet under symbolic analysis.

### A. Formal Verification and Piecewise-Linear Structure

Neural networks with ReLU activations define piecewise-linear functions whose geometric structure underpins the difficulty of formal verification. Each ReLU unit adds a potential decision boundary of the form $x_i =0$ - the combination of many ReLUs forms an exponential number of linear regions. For a network with $n$ ReLUs, the theoretical maximum number of regions is $2^n$, although practical architectures typically realize a smaller—but still extremely large—subset of them. Throughout the project, this combinatorial structure became immediately apparent when constructing the SMT encodings. Even relatively small subnetworks of MiniNet produced multiple branching points, requiring the solver to explore or prune many possible activation patterns. The papers emphasize that the challenge is not merely the number of regions but the fact that robustness verification requires proving that none of the reachable regions contradict the desired classification. My experiments aligned sharply with this theoretical intuition: stable ReLUs made the solver fast, unstable ones caused rapid blow-up.

### B. Relationship Between Local and Global Robustness

Robustness can be framed either as a global property of a model or as a local property around a particular input. In my work, verification focused on local $l_{\infty}$
-ball robustness. This differs from global robustness claims, where one tries to guarantee correct behavior for all inputs within a domain. The local setting is more tractable but still extremely challenging. The literature notes that the size of the perturbation radius $\epsilon$ is the dominant factor controlling verification difficulty, which my results echoed: when $\epsilon=0$ the verification problem collapses to a single point and becomes trivial; even small positive $\epsilon$ values introduce a high-dimensional feasible region with many potential activation flips. The phenomenon that $\epsilon =0$ solves instantly but $\epsilon >0$ may require hundreds of seconds is directly tied to the geometry of these perturbation sets.

### C. Supplementary Notes on Convolutional Layers and Unrolling

I conducted deeper exploration into the consequences of convolution unrolling on verification workload. A convolutional layer with kernel size $k \times k$, input channels $c$, and output dimensions $H \times W$ effectively becomes a large matrix multiplication of size $(H \cdot W) \times (k^2 \cdot c)$. Thus, even a modest convolution such as a $3 \times 3$ kernel with 8 channels generates substantial affine constraints when flattened into equations. This phenomenon is well discussed in the verification literature, particularly in works on abstract interpretation for CNN architectures. In practice, the expansion of constraints reduces the solver's ability to maintain tight variable bounds. During IBP experiments on MiniNet, I observed that convolutional layers significantly widened the interval bounds and sharply increased the number of predicted unstable ReLUs in subsequent layers. This confirmed theoretical predictions that unrolling convolutional layers has structural implications for all later verification stages, making CNNs substantially more difficult than dense networks of comparable parameter count.

### D. Mathematical Formulation of ReLU Constraints

To better understand solver behavior, I studied the mathematical forms of ReLU encodings across multiple verification tools. Most SMT-based approaches use a big-M style or disjunctive encoding, which in its exact form introduces two linear subcases per ReLU. If $x$ is the pre-activation variable and $y$ is the post-activation variable, the constraints are: $x \leq 0$, $y=0$ or $x \geq 0$, $y=x$. 

Some tools relax these constraints when verifying over intervals or convex abstractions, but Z3 and Marabou attempt to maintain the exact semantics. During experiments, I noticed that the adversarial region constraints combined with the affine transformations often created intermediate variables whose sign could not be conclusively determined using forward interval reasoning, forcing the challenger SMT solver to branch. The mathematical simplicity of the ReLU belies the depth of the induced combinatorial explosion.

### E. Example Symbolic Constraint Systems for MiniNet

To highlight the practical form of the constraint systems generated in the project, I captured example fragments from the Z3 encodings. Even for a single input image, the symbolic structure expanded quickly. For instance, the first convolutional block produced hundreds of linear expressions of the form:

$z_{i,j} = \sum _{p,q,c} w_{p,q,c} \cdot x_{(i+p),(j+p),c} +b$

followed by a set of ReLU constraints:

$\text{Or }((z_{i,j} \leq 0 \land a_{i,j} = 0), (z_{i,j} \geq 0 \land a_{i,j} = z_{i,j}))$

When the adverserial region was introduced, each variable gained bounds of the form: 

$x_0 - \epsilon \leq x \leq x_0 + \epsilon$

which caused the interval ranges to propagate and loosen through later affine layers. Although these constraints are individually simple, their combination created the complex search structure observed in solver logs. Reviewing these examples helped me understand why solver runtimes behaved in the patterns reported earlier.

### F. Interval Propagation and ReLU Stability Estimation

The IBP stage in the verification funnel played an important predictive role in determining which points were likely to be solver-hard. Interval propagation computes approximate bounds by repeatedly applying forward interval arithmetic:

$[h_{min}, h_{max}] = W[x_{min}, x_{max}] + b$

After computing these intervals, a ReLU is considered stable if $h_{max}\leq 0$ or $h_{min} \geq 0$ and unstable otherwise. By inspecting these results across many values of $\epsilon$, I constructed empirical curves showing that the number of unstable ReLUs increased nearly linearly with $\epsilon$, for the first few layers but very sharply for middle and late layers due to the loose bounds produced by convolutional unrolling. This provided insight into why the solver transitioned from easy to extremely hard at particular thresholds of $\epsilon$. The literature suggests that improving interval propagation tightness—through methods like quadratic relaxations or zonotope propagation—would likely reduce many of these verification costs.

### G. Observed Solver Behaviors and Benchmark Patterns

In addition to the main runtime results, I collected supplementary data from solver logs, including branch counts, number of SAT/UNSAT leaves explored, solver restarts, and measurable differences between Z3’s linear arithmetic backend versus its case-splitting heuristic. I found that Z3 would often prematurely explore low-likelihood activation patterns due to incomplete bound propagation, leading to long search paths that could have been pruned with tighter interval tightening. In contrast, Marabou’s branching heuristics were more specialized but were frequently interrupted by the API failures. These solver behaviors matched known observations in verification papers, which consistently emphasize the need for better pruning strategies and efficient bounds. While the solvers were usable, their behavior reflected the research-grade state of modern formal verification tools.

### H. Additional Notes on the Verification Funnel Design

The funnel architecture developed for this project drew from ideas present in several verification papers that combine counterexample-guided search with bound-based pruning. The idea is that most inputs do not need SMT-level checking, and that robustness can often be established with simpler methods. In practice, my funnel eliminated a significant portion of inputs at the PGD stage, a second wave at the IBP stage, and left a smaller percentage of points for Z3 (typical range seemed to be ~1-5%). Supplemental plots not included in the main body showed steep declines in the proportion of inputs reaching the SMT solver as $\epsilon$ increased. The funnel therefore served not only as a performance optimization but also as a diagnostic tool because by observing where inputs dropped out, I could better understand MiniNet’s vulnerability structure.

### I. Reflections on Implementation Choices and Potential Improvements

In reviewing the entire project, I identified several areas where improved abstractions, more advanced solvers, or tighter propagation methods could significantly reduce runtime. The papers strongly suggest that hybrid techniques, such as combining symbolic methods with α-β-CROWN bounds, would likely improve scalability. Additionally, alternate convolution encodings—such as using convolution-aware relaxation operators—could dramatically tighten bounds. My implementation focused on exact symbolic reasoning, which is the most principled but often the least scalable. Future work could explore specialized solvers like ERAN or β-CROWN, both of which make differential use of linear relaxations to guide branch decisions. 