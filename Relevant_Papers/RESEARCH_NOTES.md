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