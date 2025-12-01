# Research Notes from the Attatched PDFs

This file contains some notes that I took when I was reading through some of the papers that I thought would be relevant to my project. Sorry in advance for some of the broken english as these are pretty quickly taken notes

---

## Reluplex Paper

Reluplex Pdf: 
[1702.01135v2.pdf](1702.01135v2.pdf)

Reluplex is a scalable and efficent technique that is designed for verifiying propoerites of deep neural networks and in in the process, finding counter examples if they exist.

The motivation beihind the design comes from the application of Deep neural networks to real world safty ciritical systems. The example mentioned in the paper was the next-generation airborne collision avoidance system for unmanned aircraft - abbreviated ACAS Xu. Providing formal guaranetees about their behavior is challenging so the application of DNNs can be very challenging. 
- Verification is hard becuase DNNs are...
    - large
    - non-linear
    - non-convex (curves boht up and down with multiple peaks and valleys so can have several local minimum and maximum which can cause problems in the gradient descent calculation that most DNNS perform)
    - verification of simple properties is NP-Complete

That means that many general-purpose tools like SMT or LP solvers are inadequate.

**Reluplex:** SMT Solver for a theory of linear real arithmetic that is extended with Rectified Linear Unit (ReLU) constraints. Extension of the simplex methos which is a standard algorith for solving LP instances.

The main mechanism behind Reluplex comes from how it addresses the non-linear and non-convex nature of the traditional ReLU activation function. Most naive SMT solvers encode ReLUs using *DISJUNCTIONS* - which has the potential to lead to $2^n$ subproblems where $n$ is the number of ReLU nodes. Reluplex leverages the piecewise linear nature of ReLUs so the procedure iteratively searched for a feasible solution which allows the variables to temporarily violate bounds/ReLU semantics, but then corrects them using Pivot and Update operations. 
- Performance Improvements: Scalable bc relies on avoiding excessive case-splitting (`ReluSplit` rule). Implements the following:
    - Tigher Bound Derivation: the tighter varibale bounds whihc are dediced can sometimes eliminate ReLus entirely by fixing them to an active or inactive state without splitting
    - Conflict analysis: Allos the solver to efficently undo multiple nested splits when a contradiction is found.
        - Ex from paper: $lb(x) > ub(x)$
    - Floating Point aritmetic: Used for speed. paper does mention that it does not guarantee soundess and that would be a goal for future work.

Reluplex was evaluated on a family of 45 real-world DNNS that were developed as an early prototype for ACAS Xu. Layers are fully connected, have 8 layers, contain 300 ReLU nodes each. Reluplex was able to verify the properties of the netwroks an **order of magnitude larger** than those previously analyzed. Experiments shows that Reluplex solved all 8 simple test instance where state-of-the-art SMT and LP solvers generally performed poorly or times out. 

---
## Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks

Pdf: 
[1705.01320v3.pdf](1705.01320v3.pdf)

Key concept in this paper was the node-wise application of piecewise linear activation functions. Verifying properties of feed-forward netwroks is challenging becasue the resulting SMT instance are difficult for modern SMT solvers to handle due to the large number of node phase combinations that they must iterate through. This paper proposes an approach that would support all piece-wise linear activation functions that are used in modern network architechtures which includes the ReLU and MaxPool nodes. MaxPool nodes are crucial for feature detection but are not supported by come earlier verification methods - the paper reference a previous paper byt Katz.

The approach combines SAT solving and linear programming where the search process is guided by the SAT solver which determines the phases of the nodes. A key starting point is the addition of a global linear approximation of the overall network behavior to the verification problem. This approximation is created using classical interval arithmetic to obtian initial bounds, which are the iteratively refind by solving and LP instance (inclusing the problem specification $\psi$) to derive tighter bounds. This approximation allows the sovler to quickly rule out large parts of the search space. 

Specialized Search Enchancement Procedures:
- Irreducible, Infeasible Subset Finding: When a partial node phase assignment leads to an inffeasible LP, elastic filtering is used to identify minimal infeasible constraint sets which allows the genertation of shorter conflict clauses for the SAT solver.
    - elastic filtering is the process of dynamically narrowing doen massive datasets based on specific criteria or conditions without compromising performance. Allows  the refinement of search results by applying rules that determine which documents to include or exclude. Typically yes/no method rather than ranking document relevance scores
- Implied Node Phase Inference: When a partial assignment is feasible, a specialized optimization function is used during the linear programming solving to minimize the error which yields tight assignments for some nodes. In order to make the search more efficent, these inferred phases can be cached and retained.
- Detecting Implied Phases: Similar to unit propogation in classical SAT solvers, implied node phases can be detected by propgating lower and uppder bounds thorugh the network which is considerable faster than calling the LP solver
- Planet: this is the tool that is used ot implement the approach. Stands for Piecewise Linear feed-forward neural network verification tool and is a tool written in C++ that incorporates the GLPK (GNU Linear Programming Kit - solve large scale linear programming problemsinternally in the search space) and a modified verion of Minisat SAT solver which handles the Boolean part of the verification problem.
    - Experiments on collision avoidance and handwritten digit recognition (MNIST dataset) case studues demonstrated that Planet often achieved faster computation times compared to external solvers (paper specifically compares to Yices and Gurobi), especially when those external solvers did not utilize the linear approximation constraints that were developed/discussed in this paper.
    - [Planet Github Repo](https://github.com/progirep/planet)

---
## Using Z3 for Formal Modeling and Verification of FNN Global Robustness

Pdf: 
[2304.10558v2.pdf](2304.10558v2.pdf)

THis paper talks about the research gap in global robustness analysis of Feed Forwar Neural Networks (FNNs). Emphasize that most existing verification techniques focus only on local robusteness against small perturbations near specific datapoints. 
- **Perturbation:** deviation of a system, moving object, or process from its regular or normal state or path, cause by outside influence (via Google)

The paper proposes a complete specification and implementation of the DeepGlobal framework using hte SMT sovler Z3. DeepGlobal is designed to identify all possible Adverserial Dangerous Regions (ADRs), thereby analyzing global robustness. 
- ADRs are specific areas in a machine learning model's input space where inputs are highly susceptible to small, ofter inperceptible, adverserial perturbations. Can cause the model to take drastically incorrec or unintended predictions. Indicate areas where a models decision boundaries are sharp or non-robust whihc makes the model fragile to minor changes in input data. When these regions are explicitly presented, engineers can be warned and alerted to potential security weaknesses in systems, in this case networks. Can help to guide adverserial training to aid in making the model more robust

DeepGlobal [(Paper)](https://link.springer.com/chapter/10.1007/978-3-030-91265-9_2)
introduces the Sliding Door Network (SDN) architecture which uses the sliding door activation function (sda) which is a function that groups neurons and designates both an active and an inactive door, leaving others as trivial. The active door is multiplied by $/alpha >1$ and the inactive foor is multiplied by 0. This drastically redices the number of possible activation patterns when compared to more standard activation functions like ReLU. The groups are equally sized and the behavior of all the neurons in the group are linked (meaning they all behave in the same general way either all positive or all negative). At each layer, send neurons to doors.
- Classic ReLU based FNN: the number of activation patterns is $O\left (\prod 2^{n_i} \right)$
- SDNs: $O\left (\prod {n_i}^2 \right)$

The paper does talk about formal Z3 specifications for FNNs, SDNs, Activation Patterns (referred to as $C_{AP}(A)), and Adverserial Dangerous Regions. The links to them are in the paper. They mention that the verification process involves enumerating all valid APs and solving the necessary constraints using the Z3 solver.

A key imporovements made in this paper is incorporating a meaningful conditon into the ADR definintion, as a method of refinement, and recognizing that simply finding a decision boundary is insufficent if the samples are "rubbish" (sorry that word choice is funny had to include). In order to define a meaningful region, they train auto encoders to generate prototypes ($P_i$) for each class that are based on the average encoded code of samples in that class. A sample *y* is considered meaningful if it is constrained to be near the prototype. THey used the following: $|y-P_i|_p \leq r$. The authors also improved/optimized the implementation efficency by customizing the Z3 tactics `simplify`, `normalize-bounds`, and `solve-eqs`.

---

## The Marabou Framework

Pdf: 
[KHI+19.pdf](KHI+19.pdf)

Marabou is a framework that was build upon the preceding Reluplex projects and is designed for the verification and anyalsis of DNNs.
It is an SMT-based tool that transforms DNN verification quereies (involving boht linear and non=linear constraints) into constraint satisfaction problems. It uses and SMT based lazy search technique (similar to what was presented in the Reluplex paper) that aims to solve the linear constraints first and treat the non-linear constraints (like activation functions) lazily. Additionally, it expands on Reluplex by providing native support for arbitrary piecewise linear activation functions, including the RELU function and the Max function ($max(x_1, x_2,...x_n)$). Also supports both fully connects and convolutional DNNs.

Improvements/Enchancementsover Reluplex:
- Includes a custom-built simplex-based linear programming core, replacing the external GLPK solver. This eliminates communication overhead and proved the felxibility necessary for tightly integrating the solver with DNN verification.
- Supports deduction that is based on the network topology - symbolic bound tightening - and this reasoning is crucial for curtailing the search space.
    - symbolic bound tightening is a method used in optimization and causal inference to find closed-form, analytical expressions for the teghtest possbile bounds on variables or causal effects. Instead of numerical computation on specifc data, it generates general expressions that are valid across all compatible data, providing a more complete understanding of the range of possible values.
- Uses whats called a Divide and Conquer Mode (D&C) that partitions the input query region into simpler sub-queries if an initial timeout is reached. This mode is naturally suited for parallel execution but can still yield significant speed-ups weven when run sequentially.
- Flexible interfaces as it accepts queries via multiple input formats including textual formats and protocol buffer files generated by TensorFlow

The core engine of Marabou iteratively atempst to satisfy constrains, prioritixing fixing violated linear constraints (via simplex steps) and then fixing violated piecewise linear constraints. Non linear constraints (think ReLU and Max) are represented by abstract classes that define methods for checking satisfaction, proposing fixes, returning equivalent linear constraints for case-splits, and returning entailed bounds tightenings. Generally outperforms Reluplex and Planed on common benchmarks, not ReluVal. But when using 64 cores, Marabou was able to close the gap and sometimes outperform ReluVal.

---

## An Abstraction Domain for Certifying Neural Networks

Pdf: 
[DeepPoly.pdf](DeepPoly.pdf)

This paper presents an novel method and system that they call **DeepPoly** that was created for scalable and precise certification of DNNs. The primary goal is to address teh challenge of verifying DNN properties, such as rpbustenss against adverserial attacks, which oftern requires an analyzer that still remains precise even when scaling to larger networks. Their key technical insight is a new abstract domain that is specifically tailored for neural netwroks that combines floating point polyhedra with intervals - allows for the careful balancing of analysis scalability and precision.
- [Floating point polyhedra](https://www.researchgate.net/publication/47652961_A_Sound_Floating-Point_Polyhedra_Abstract_Domain) is a mthod in static program analysis that is used to determine properties of programs that use floating point numbers, such as their potential range of values. Like the name suggests, it uses polyhedra, which are geometirc shapes defined by linear inequalities, to represent program variables, and a sound adaptation of this representation to handle floating point arithmetic, including its precision loss and potential errors. Makes it more practical and efficent for analyzing large programs than traditional methods
    - Uses the idea of an abstract domain which like we discussed in class is a set of abstract values that represent a collection of concrete program states.

Notes on DeepPol's Abstract Domain and Methodology:

The abstract domain: $A_n$ over $n$ variables is represented as a tuple $\langle a^{\leq}, a^{\geq}, l, u \rangle$
- Polyhedral Constraints ($a^{\leq}, a^{\geq}$): Each variable $x_i$ is associated with a lower and an upper polyhedral constraint. These constraints, crucially, only related to $x_i$ variales that appear earlier in the netwrok - so those with smaller indices
- Concrete Interval Bounds(l, u): The domain tracks a concrete lower bound ($l_i$) and upper bound ($u_i$) for each variable $x_i$ ensuring that the interval $[l_i,u_i]$ OVERapproximates the set of values $x_i$ can take - so a domain invariant. 
- The domain restricts the overall number of conjuncts to $2n$ which is necessary to avoid the exponential blowup that occurs if you were to support the full expressive power of convex polyhedra on netwroks with thousands of neurons.

DeepPoly employs specialized abstract transformers designed for common neural netwrok function in an effort to maintain soundess and efficency:
- Affine transformations compute concrete bounds ($l'$ and $u'$) be performing backsubstitution. Kind of similar to the concept of backpropogation. This recursively replaces polyhedral constraints until the bounds only depend on the input variables. While this process is computationally expensive, it is necessary for precision. 
- ReLU Activation: For Reul ($x_j:= max(0,x_i)$) when the input bounds $l_i$ and $u_i$ span zero (meaning that $l_i < 0$ and $u_i > 0$), the transformation is inexact. DeepPoly therefore approximates the ReLU assignment using the thghtest convex hull approximations that mainatins only one lower bound and one upper bound constraint. This choice is deliberate and made to prevent the blowup caused by introducing multiple lower relational constraints. If the bounds do not span 0 - so  $l_j \ge 0$ and $u_j \le 0$ - then the transformer is exact.
    - Convex hull approximation is a subset of points that approximately repressents the smallest convex shapre containing a larger set of points. This provides a faster and more memory efficent solution when exact calculations are too complex or too time-consuming. Algorithms typically simplify the problme by projecting the points onto a grid and selecing points from each cell. Useful for applications were some loss of accuracy is acceptable (examples given are data compression and image recognition)
- Custom abstract transformers are provided for sigmoid, tanh and maxpool functions
- The domain and its transformers are modified using the interval linear form, and floating point interval arithmetic ($\oplus _f, \ominus _f$, etc.) to ensure soundess even under floating point arithmetic which is used heavily in nerual networks. 

DeepPoly is used for robustenss certification which involves proving that all inputs within a specified adverserial range classify to the same label. The analysis is performed by propagating the abstract input region through the network's layers using the abstract transformers.

DeepPoly is also capable of verifying complex specifications beyond the standard $L_{\infty}$ -norm perturbations. These attacks are paramaterized by a constant $\epsilon$ where the adverserial region contains all the perturbed images $x'$ where each pixel $x'_i$ has a distance of at most $\epsilon$ from the corresponding pixel $x$ from the original input. The paper used different values of $\epsilon$ when evaluating robustness.
- First work shown to verify robustness when the input image is subjected to complex geometric transformations like rotations that employ linear interpolation. The two methods Google mentioned were linear and spherical
    - Linear (Eulers angles): Interpolate each component of the roatation (pitch, yaw, roll) over time. This is simple to implement but the path of rotation is undesirable (often rotates in unexpected ways) and does not result in a constant angular velocity.
    - Spherical (Slerp): Interpolate between two rotations using a specialized formula - most commmonly quaternions which is best explaned as thinking of representating rotations in a 4D space. This interpolates along the shortest path between the two rotations, results in a constant angular velocity, avoids singularities (unlike above) and provides a smooth and natural looking rotation. However it is more computationally intensive and requires converting to and from different rotation representations.
        - $Slerp(q_1,q_2;u) = q_1(q_1^{-1}q_2)^u$
        - $q_1$ and $q_2$ are the initial and final quaternions and $u$ is the interpolation parameter and ranges from 0 to 1
- To handle the inprecision caused by large angular ranges, DeepPoly uses a form of abstraction refinement that is based on trace partitioning. This means it involves subdividing the rotation angle into interval batches, computing the smallest common "bounding box" AKA region for each batch using interval analysis and then running the neural network analysis on these smaller, more precise regions.

Overall this system implemetns a complete and parallelized verstion that was shown experimentally to be more precise that prior stat of the art tools like AI $^2$, Fast-Lin and DeepZ. Also maintains scalability particularly when applied to feed-forward networks. Also reduces the percentage of hidden units for which the ReLU transformer is inexact (hence the better precision. )

---

## Abstraction-Based Proof Production in Fomral Verification of Neural Networks

Pdf: 
[2506.09455v1.pdf](2506.09455v1.pdf)

This paper talks about preliminary work that addresses the critical gap between achieving scalability in DNN verfication and mainitaing reliability through formal proof production. It lays the foundation for a novel framework that is designed to integrate abstraction nechniques (which are vital for overcoming the NP-complete complexity) directly into the formal proof generation pipeline - for a researcher focused on neural network verification.

The proimary motivation the authors mentions is that while abstraction improves the scalability of a netwrok through simplification, the existing proof verification tools that are out there do not support abstraction based reasoning. Additionally, relying soleley on verification algorithms is insufficent, as errors in their implementation can compromise the soundess of the networks. Therefore, by generating formal proofs that can be checked by an independent program, the reliability of teh verifiers correctness can be enchances. 

There are two main challenges mentioned:
1. Enabling proof production for abstraction based solvers
2. Generating more compact proofs - verifiying a smaller abstract netwrok is typically faster and can lead to a potentially significant reduction in proof size

The key concept that this paper introduces is the absrtract proof - which follows a modular structure that seperates the verfication challenge into two INDEPENDENT and verificable components: The Verification Proof for the Abstract Network and the Proof of Abstraction Correctntess. The verification proof demonstrates that the required specification (in the paper they use property $Q$) holds in the simplified abstract network $\hat{f}$. The proof of correctness, on the other hand, is essentially trying to show that its an over approximation. It is a fomral proof that the abstraction soundly approximates the original network $f$. This ensures that if the property holds for the abstract netwrok, then it is mathematically guaranteed to hold for the original netwrok as well. That is: $unsat(\langle \hat{f}, P,Q \rangle) \implies unsat(\langle f,P,Q \rangle)$. 

As an bonus, this modularity makes the framwork agnostic to the specific DNN verifier and abstraction technique used (provided that the tools can generates the corresponding proof components that is).
However, in the paper, the authors nots that they use two established verification tools to demonstrate their proposed framework.
- For the verifier, they used Marabou, specifically the proof-producing version.
- And for the abstraction engine, they used [CORA](https://www.researchgate.net/publication/283567778_An_Introduction_to_CORA_2015) which uses reachability analysis and an abstraction-refinement extenstion.

The papers novel contribution is that it proposes the first method for generating formal proofs for the abstraction process itseld. The CORA abstraction method works by merging neurons with similar bounds to reduce the size of the netwrok. This results in an abstract network that outputs a SET of vectors that captures the deviation from the original network using interval biases. The proof scheme is depicted using rules like `triv-abs`, `base-abs`, and `lk-abs` which are used to establish the correctness of the neuron merging process layer-by-layer. The core principle is proving containment: if Property 1 (Neuron merging) is applied to obtain an abstract network $\hat{f'}$, then it must hold that $\forall x \in P: \hat{f} (x) \subseteq \hat{f'} (x)%$. This formally establishes that any subsequent abstract model SOUNDLY over-approximates the previous one which yields a complete proof of over-approximation for the entire network.. 

Secondly, the paper talks about rpoving the abstract correctness component using Marabou. Since the resulting abstract network from CORA generalizes standard DNNs by having interval biases (in contrast to having scalar biases) and outputting a set of vectors, the verification query $\langle \hat{f},P,Q \rangle$ must be adapted to Marabou. The paper proposes two methods to encode the interval biases.
1. Linear Inequalities: Express the linear constraints of the abstract netwrok directly as a pair of linear inequalities that reflect the lower and upper obounds of the abstracted bias interval
2. Skip Connections: Introduce a fresh input variable for each boas term, connected via a weight-1 skip connection the relevant neuron, and extending the input property (P) to enforce the bias interval bounds on this new variable. 
    - A weight-1 skip connection is justa varaint of the standard skip connection, which adds a layer's input directly to its output, where the weight of the path is 1 making it an Identity path.

Both of these methods allow the query to be verified directly by Maraboue which enables the successful generation of the verification proof.