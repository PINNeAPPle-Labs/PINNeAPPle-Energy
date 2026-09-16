# Physics AI for Energy: A State-of-the-Field Reference

*Scope: physics-informed / scientific machine learning (SciML) applied to electricity generation, transmission, and distribution — power grids, wind, solar, nuclear, energy storage, and power electronics. Oil & gas subsurface applications are intentionally excluded (covered by a separate track).*

*Compiled September 2026. This document is meant to seed an open-source reference repository; entries are drawn from real papers, tools, and companies found via literature and web search. Where evidence was thin, that is noted explicitly rather than filled in.*

---

## Table of Contents

1. [Why Physics AI for Energy](#1-why-physics-ai-for-energy)
2. [Academic Research Landscape](#2-academic-research-landscape)
3. [Model Architectures by Application](#3-model-architectures-by-application)
4. [Commercial Products & Startups](#4-commercial-products--startups)
5. [Numerical Methods & Solvers (Baselines and Hybrid Targets)](#5-numerical-methods--solvers-baselines-and-hybrid-targets)
6. [Geometry & Mesh Resources](#6-geometry--mesh-resources)
7. [Datasets & Data Generators](#7-datasets--data-generators)
8. [Industry Standards & Validation Context](#8-industry-standards--validation-context)
9. [Open Challenges & Gaps](#9-open-challenges--gaps)
10. [Roadmap: Building the Reference Hub](#10-roadmap-building-the-reference-hub)
11. [Sources](#11-sources)

---

## 1. Why Physics AI for Energy

The energy system is undergoing two simultaneous transitions that make physics-informed ML unusually valuable: (1) rapid growth of inverter-based, weather-driven generation (wind, solar) that is harder to forecast and control than dispatchable thermal plants, and (2) a wave of new nuclear (SMRs, advanced reactors) and storage deployment that needs faster, cheaper, and still-trustworthy simulation than legacy high-fidelity codes allow. Pure black-box ML often fails on grid, battery, and reactor problems because these are safety- or reliability-critical systems governed by well-known conservation laws (Kirchhoff's laws, Butler-Volmer kinetics, Navier-Stokes, neutron transport) — physics-informed approaches aim to keep the speed of ML while inheriting the guarantees of the underlying physics.

---

## 2. Academic Research Landscape

### Power systems / grid
- **Priya Donti** (MIT EECS/LIDS, previously CMU) — leads a group on ML for forecasting, optimization, and control in high-renewables grids; work on physics-informed machine learning with optimization-based guarantees for AC power flow (*Applied Energy*, 2023) and "DC3"/differentiable-optimization approaches to enforcing hard constraints in learned OPF solvers. Also a founder of Climate Change AI.
- **Spyros Chatzivasileiadis** (DTU) — co-author of the landmark early PINN-for-power-systems paper "Physics-Informed Neural Networks for Power Systems" (Misyris, Venzke, Chatzivasileiadis, arXiv:1911.03737, 2019/2020 PES General Meeting), which applied Raissi-style PINNs to swing-equation dynamics and showed order-of-magnitude speedups over numerical ODE solvers for transient stability prediction. This is widely cited as the field's founding paper.
- **Steven Low** (Caltech) — foundational work on convex relaxations of optimal power flow (semidefinite/second-order-cone relaxations), which underlies much of the "physics-informed" OPF-learning literature that uses these relaxations as tractable physical constraints.
- **Line Roald** (University of Wisconsin–Madison) — power system optimization under uncertainty, chance-constrained OPF, learning-augmented grid operations.
- **Pascal Van Hentenryck** (Georgia Tech, NSF AI Institute for Advances in Optimization) — OPFLearn, DeepOPF-style end-to-end learning for AC-OPF, and constraint-embedding neural architectures.
- **Deepjyoti Deka** (Los Alamos National Lab) — graph-based/graphical-model learning for grid topology estimation and physics-informed GNNs for state estimation.
- IEEE PES has an active "Machine Learning for Power Systems" community (workshops since 2021) and a "Physics-informed Machine Learning in Power Systems" trending-tech page tracking the subfield.

### Wind energy
- **NREL National Wind Technology Center** (Paul Fleming, Jennifer King, and collaborators) — developers of **FLORIS** (FLOw Redirection and Induction in Steady-state), the leading open-source engineering wake model for wind farm control research, co-developed with **TU Delft**; also maintainers of SOWFA and the newer **AMR-Wind** LES code.
- **DTU Wind Energy** (Denmark) — co-developer of the IEA 15 MW reference turbine (with NREL), HAWC2 aeroelastic code, and active wake/CFD research groups; frequent OpenFAST/HAWC2 cross-validation studies (e.g., comparison of loads from HAWC2 vs. OpenFAST on the IEA 15MW turbine).
- Groups publishing neural-operator wake models (2024–2026): Fourier Neural Operator and PINN hybrids for floating offshore turbine wake dynamics under platform motion (arXiv:2604.23937, "Multi-scale Dynamic Wake Modeling of Floating Offshore Wind Turbines via FNOs and PINNs"); Graph Neural Operator approaches for whole-windfarm wake flow (*Wind Energy Science* preprint, wes-2025-261); channel-assisted FNOs for real-time yawed-turbine far-wake prediction.

### Battery / energy storage
- **Richard Braatz** (MIT, Edwin R. Gilliland Professor) — machine-learning-based prediction and classification of Li-ion battery cycle lifetime, optimal fast-charging protocol design; co-led (with Stanford's William Chueh and colleagues, plus Toyota Research Institute) the widely cited 2019 *Nature Energy* early-prediction-of-battery-lifetime study that produced the "Toyota/Stanford/MIT" battery dataset.
- **Simona Onori** (Stanford, Department of Energy Science & Engineering, director of the Stanford Energy Control Lab) — battery state-of-health estimation combining control theory, electrochemistry, and data-driven observers; active in physics + ML hybrid state estimators.
- **Kandler Smith** and colleagues at **NREL** — physics-based Li-ion degradation modeling (aging/SEI growth models) frequently used as the "physics" backbone for hybrid ML degradation models.
- P2D/SPM PINN literature (2024–2026): Fourier-enhanced DeepONet benchmarks against conventional PINNs for the single particle model, showing ~10x speedup vs. numerical solvers (arXiv:2601.01466); transfer-learning PINNs for single-particle-model-with-electrolyte (SPMe) state estimation across battery chemistries (arXiv:2606.28220); parameter estimation of electrochemical models via PINNs (*Journal of Power Sources*, 2024).
- **Venkat Subramanian** (University of Washington) and **Gregory Plett** (University of Colorado Colorado Springs) are established figures in reduced-order/physics-based battery modeling (SPM, single-particle-model variants) whose formulations underpin many PINN implementations, though direct PINN authorship should be verified per-paper.

### Nuclear
- **CTF4Nuclear** — the first open "common task framework" benchmark platform for ML surrogate modeling and state estimation across nuclear systems (Molten Salt Fast Reactor, Heat-Pipe Microreactor, and others), enabling apples-to-apples comparison of surrogate architectures (arXiv:2605.15549, 2026).
- Active research on hybrid ML for **critical heat flux (CHF)** prediction inside the **CTF** subchannel thermal-hydraulics code (multiple 2024–2026 arXiv/OSTI papers, including uncertainty-quantified hybrid CHF models combining physics correlations with DNN/Bayesian corrections).
- PINNs applied directly to reactor physics: neutron diffusion eigenvalue problems (uncertainty analysis of data-enabled PINNs, arXiv:2303.08455) and reactor safety analysis PINNs (*Annals of Nuclear Energy*, 2025).
- Deep-learning surrogates for severe-accident modeling built on the **ASTEC** code (arXiv:2607.04450) and for **TRACE**-based transient simulation.
- National labs (INL, ORNL, ANL) run dedicated digital-engineering/AI programs for advanced reactor design under DOE's Advanced Reactor Demonstration Program, though public peer-reviewed output is less centralized than academic groups.

### Key venues
- **Power Systems Computation Conference (PSCC)**
- **IEEE PES General Meeting**, IEEE Transactions on Power Systems / Smart Grid / Industrial Informatics
- *Applied Energy*, *Nature Energy*, *Journal of Power Sources*, *Wind Energy Science* (Copernicus/EGU), *Annals of Nuclear Energy*, *Nuclear Technology*
- NeurIPS/ICML "Climate Change AI" and "ML4Physics" workshops increasingly carry grid/battery/wind papers.

---

## 3. Model Architectures by Application

| Domain | Architecture patterns in use | Representative examples |
|---|---|---|
| Battery electrochemistry | PINNs solving SPM/SPMe/P2D PDE systems (solid/electrolyte diffusion, Butler–Volmer kinetics, Ohm's law); DeepONets (incl. Fourier-enhanced) for operator learning across current profiles; transfer-learning PINNs for cross-chemistry generalization | Fourier-DeepONet SPM surrogate (~10x speedup vs. numerical solver); SPMe-PINN with transfer learning for SOC/SOH estimation |
| Wind wake / farm flow | Fourier Neural Operators (FNO) and PINN hybrids for unsteady wake fields; Graph Neural Operators/GNNs treating each turbine as a node with message-passing for wake interaction; channel-assisted FNOs for real-time far-wake prediction; transformer-based operators for terrain-influenced wind fields | FNO+PINN for floating turbine wake under platform motion; GNN wake models achieving RMSE ~0.35 m/s on held-out layouts |
| Grid state estimation | Physics-informed GNNs embedding branch admittance/topology into message passing; hierarchical GNNs for low-observability distribution substations; optimization-embedded ("layer") end-to-end estimators | Physics-informed graphical NN for state estimation (>20% MSE reduction vs. conventional WLS on IEEE test systems, *Applied Energy* 2024) |
| AC/DC power flow & OPF | Physics-informed loss functions enforcing power-flow equations and operational constraints without labeled training data; typed/heterogeneous GNNs representing buses/lines/generators as distinct node/edge types; differentiable-optimization layers (DC3-style) for hard-constraint satisfaction | "Optimal Power Flow with Physics-Informed Typed Graph Neural Networks" (IEEE Trans., 2024); GNN-initialized AC-OPF warm-starts for interior-point solvers |
| Nuclear thermal-hydraulics | Hybrid ML+correlation models for critical heat flux inside CTF; deep-learning surrogates for TRACE/ASTEC transients; PINNs for neutron diffusion eigenvalue problems; Bayesian-ensemble UQ for safety-relevant quantities | CTF4Nuclear benchmark suite; Bayesian-optimized deep ensembles for sodium fast reactor thermal stratification (arXiv:2412.08776) |
| Turbine/plant digital twins | Physics-ML hybrids combining reduced-order structural/aero models with real-time sensor assimilation; NVIDIA PhysicsNeMo-based neural surrogates replacing CFD in the loop | Siemens Energy grid-asset simulation reported at ~10,000x speedup using NVIDIA PhysicsNeMo (formerly Modulus) |
| Forecasting (load/renewables) | Time-series foundation models adapted to power systems: Chronos/Chronos-2, TimesFM, Moirai, MOMENT, TimeGPT-1, Tiny Time Mixer — evaluated zero-shot and fine-tuned against domain covariates (temperature, calendar effects) | Empirical benchmarking of TSFMs for power-system forecasting (arXiv:2604.22077); covariate-informed grid load forecasting with foundation models (arXiv:2609.06656) |
| Grid control | Deep reinforcement learning for voltage/frequency regulation, grid-forming converter control, islanded microgrid restoration; safe-RL variants with Lyapunov/constraint guarantees; multi-agent and graph-RL formulations | "Reinforcement Learning Meets the Power Grid" survey (safety + multi-agent emphasis); "A Review of Safe RL Methods for Modern Power Systems" (arXiv:2407.00304); Graph RL for power grids survey (arXiv:2407.04522) |

General cross-cutting patterns worth noting for the repo's architecture taxonomy:
- **PINN (loss-based)**: physics as a soft penalty in the training loss (PDE residual, conservation law violation).
- **Neural operators (DeepONet, FNO, GNO)**: learn a mapping between function spaces, well suited to varying boundary/inflow conditions (turbine layouts, atmospheric states) without retraining.
- **Graph neural networks**: natural fit for grid topology and turbine-array interactions where the physical connectivity is itself the inductive bias.
- **Hybrid/hard-constraint layers**: differentiable optimization or projection layers that guarantee feasibility (e.g., DC3, OPF-Learn) rather than merely penalizing infeasibility.
- **Foundation models**: pretrained time-series transformers repurposed for load/renewable forecasting via fine-tuning or zero-shot + covariates.

---

## 4. Commercial Products & Startups

**Simulation & digital-twin platforms**
- **NVIDIA PhysicsNeMo** (formerly Modulus) + **Earth-2** + **Omniverse** — physics-ML framework and digital-twin stack applied to energy; Siemens Gamesa/Siemens Energy used Modulus+Omniverse for wind-farm wake digital twins (reported up to 4,000x speedup vs. LES) and Siemens Energy separately reported ~10,000x speedup on power-grid asset simulation using PhysicsNeMo. Earth-2 is also used for solar irradiance forecasting (AWS Batch integration for near-real-time energy production forecasts) and nuclear plant digital-twin work.
- **Ansys** (now part of Synopsys, post-2025 acquisition) — Ansys Fluent/Mechanical for wind turbine blade aero-structural design, gas turbine blade simulation; broad simulation suite used as physics baseline/training-data generator for surrogate ML.
- **DNV Bladed** — the leading commercial wind turbine design/certification simulation platform ("Bladed 5"), interoperating with OrcaFlex, ANSYS, MoorDyn, and DNV's Sesam/SACS for floating/offshore structural analysis; used as ground truth for many ML wake/load surrogate papers.
- **Siemens Energy** — Omnivise platform for power-plant AI/digital twin and asset management; Siemens completed its acquisition of **Altair Engineering** (March 2025) to fold simulation/HPC/AI into the Xcelerator platform.
- **GE Vernova** — industrial digital twins for power generation assets; launched **GridBeats: EnergyAPM** (March 2024), combining physics-based digital models with real-time/historical data for equipment health monitoring and failure prediction.
- **Cognite** + **AVEVA** (AVEVA owned by Schneider Electric; merged with Cognite in 2024) — unified industrial data/AI platforms (Cognite Data Fusion, AVEVA CONNECT) used across power generation and grid operators for asset performance and simulation-data contextualization.
- **Monolith AI** (UK) — no-code ML platform for engineering simulation surrogate modeling, marketed for automotive/aerospace but also applied to energy hardware testing; independent verification of energy-specific case studies was not found in this research pass and should be checked directly with the vendor before citing specifics.

**Grid / DERMS / demand-side AI**
- **C3.ai** — enterprise AI platform with energy-sector deployments (predictive maintenance for utilities); partnered with Shell and Microsoft on the "Open Energy AI Initiative," an open marketplace for energy AI applications.
- **AutoGrid** — DERMS and virtual-power-plant AI/ML software (founded 2011, Redwood City); acquired by **Uplight** (deal announced Dec 2023, closed early 2024) after being divested by Schneider Electric. AutoGrid Flex remains an AI-driven DERMS product line.
- **Uplight** — AI-driven demand-side management and DERMS, now incorporating AutoGrid's technology.
- **Verdigris Technologies** — machine-learning circuit-level energy monitoring; partnered with **ABB** to predict unplanned power-consumption peaks.

**Battery ML startups**
- **Aionics** (Palo Alto, founded 2020 by Stanford PhDs) — AI/materials-informatics for electrolyte and battery-material discovery; expanding from EV into eVTOL/aviation battery applications.
- **Voltaiq** (Berkeley, founded 2012) — Enterprise Battery Intelligence platform automating battery test-data collection/analytics across development, manufacturing, and field use.
- **BattGenie** (Seattle, founded 2017) — physics-based-model-driven battery management software claiming >100% cycle-life improvement via optimized charging without hardware changes.

**Nuclear digital engineering**
- **TerraPower** — uses NVIDIA Omniverse (via integration partner **SoftServe**) for an AI-powered digital-twin platform compressing plant-design iteration (geotechnical modeling, grid interconnection analysis, site layout optimization) from years to weeks for its Natrium reactor program.
- **X-energy** and **Kairos Power** — advanced reactor developers (TRISO-fueled Xe-100 and molten-salt-cooled KP-FHR respectively); Kairos signed the first corporate SMR purchase agreement with Google (Oct 2024). Public detail on their internal ML/surrogate-modeling stacks is limited; their digital-engineering approaches are less publicly documented than TerraPower's and should be sourced directly if cited in depth.

*Caveat for the repo maintainers:* several vendor claims above (speedup multipliers, ROI percentages) come from vendor blogs/press releases rather than independent peer review — flag these clearly as vendor-reported in the final repo rather than presenting them as validated benchmarks.

---

## 5. Numerical Methods & Solvers (Baselines and Hybrid Targets)

- **Power flow / grid**: Newton-Raphson and fast-decoupled AC power-flow solvers; DC power-flow linearization; **pandapower** (open-source Python, >500,000 downloads, Fraunhofer IEE) for power flow, OPF, state estimation, short-circuit (IEC 60909), and topological analysis; **PyPSA** (Python for Power System Analysis) for capacity-expansion and sector-coupled energy-system modeling; commercial tools like PSS/E, DIgSILENT PowerFactory, PowerWorld.
- **Electromagnetic transients (EMT)**: **EMTP** (Electromagnetic Transients Program, the historical reference EMT solver, now EMTP® commercial product) and open variants (ATP) for insulation coordination, switching transients, and inverter-dominated-grid transient studies — an increasingly important baseline as ML is used to accelerate EMT simulation for grids with high inverter-based-resource penetration.
- **Wind CFD**: **OpenFOAM** with actuator disk (ADM) and actuator line (ALM) models; **SOWFA** (NREL's Simulator fOr Wind Farm Applications, built on OpenFOAM, uses LES for atmospheric boundary layer + ALM turbine representation, historically coupled with the aeroelastic solver **FAST**); **AMR-Wind** (NREL's newer adaptive-mesh-refinement LES code, exascale-oriented successor to SOWFA); **FLORIS** as the fast engineering-wake-model baseline against which ML wake surrogates are typically benchmarked; **HAWC2** (DTU) and **OpenFAST** (NREL) for aero-hydro-servo-elastic whole-turbine simulation.
- **Battery simulation**: **COMSOL Multiphysics** (general finite-element multiphysics, widely used for P2D and thermal-electrochemical coupled battery models); **DandeLiion** (a specialized, extremely fast Newman-model/P2D solver offered as a cloud service, arXiv:2102.06534) explicitly built to be fast enough to serve as a "ground truth" generator for ML surrogates; PyBaMM (Python Battery Mathematical Modelling, Oxford-based open-source) is the standard open-source P2D/SPM implementation used across much of the PINN battery literature (should be added explicitly in the repo's tool list).
- **Nuclear thermal-hydraulics**: **RELAP5**, **TRACE** (NRC's system-level T-H code, successor lineage to RELAP5/TRAC), **CTF** (subchannel-level T-H code, widely used as the CHF/ML hybridization target), **ASTEC** (severe accident code, IRSN/GRS) used as the physics backbone for the deep-learning severe-accident surrogate work cited above.
- **Structural/turbine blade FE**: Ansys Mechanical, and open blade-structural tools bundled with NREL's reference-turbine repositories (e.g., blade structural properties provided alongside the IEA-15-240-RWT geometry).

---

## 6. Geometry & Mesh Resources

- **NREL 5 MW Reference Wind Turbine** — a conceptual (non-physical) offshore turbine: 126 m rotor diameter, 90 m hub height, 3-blade upwind, variable-speed collective-pitch, cut-in/rated/cut-out wind speeds of 3/11.4/25 m/s. The most widely used open reference turbine in the wind-CFD and control literature for over a decade; used as the standard OpenFAST/SOWFA validation case.
- **IEA 15 MW Reference Wind Turbine (IEA-15-240-RWT)** — jointly developed by **NREL and DTU**, publicly maintained on GitHub (`IEAWindSystems/IEA-15-240-RWT`); includes full OpenFAST and HAWC2 model files, supports both fixed-bottom and the **VolturnUS-S** semisubmersible floating platform reference design. NREL's **ROSCO** open-source controller is the standard companion controller. This is now the de facto reference geometry for large-offshore-turbine ML/CFD studies (superseding the 5 MW turbine for offshore-scale work).
- **Battery cell geometry** — no single dominant open geometric standard equivalent to the wind reference turbines was found; instead, the field standardizes around specific commercial cell form factors used in published datasets (e.g., 18650 cylindrical cells in the NASA dataset, pouch cells in the Oxford dataset, A123 APR18650M1A cylindrical cells in the Toyota/Stanford/MIT dataset) plus open cell-parameter sets distributed with **PyBaMM** (e.g., Chen2020, Marquis2019 parameter sets approximating specific commercial cells).
- **Nuclear reactor core geometry benchmarks** — OECD/NEA and IAEA maintain long-standing reactor-physics benchmark problem sets (e.g., PWR/BWR pin-cell and assembly benchmarks); the newer **CTF4Nuclear** common-task framework packages geometry + boundary conditions for several distinct reactor types (Molten Salt Fast Reactor, Heat-Pipe Microreactor) specifically for ML benchmarking, and is the most ML-native geometry/benchmark resource identified in this pass.

---

## 7. Datasets & Data Generators

**Wind & solar resource**
- **NREL WIND Toolkit** — meteorological + simulated turbine power data for >126,000 sites across the continental US, 2007–2013, distributed via the **Open Energy Data Initiative (OEDI)** data lake on AWS Open Data.
- **National Solar Radiation Database (NSRDB)** — serially complete hourly/half-hourly GHI/DNI/DHI irradiance and meteorological data, also via OEDI/AWS.
- **OEDI** overall — DOE/NREL-run open data lake spanning solar, wind, geothermal, buildings, and marine energy; grew data access by a reported 7,296% over five years, reaching 2.28 petabytes by 2024.

**Battery**
- **NASA Li-ion Battery Aging Dataset** — commercial 18650 cells (2 Ah rated), multiple aging/impedance/charge-discharge profiles; one of the earliest and most widely used public battery ML datasets.
- **Oxford Battery Degradation Dataset** — 8 Kokam 740 mAh pouch cells cycled at 40°C under Artemis urban drive-cycle profiles.
- **Toyota/Stanford/MIT ("Toyota Research Institute") Dataset** — 124–373 LFP/graphite A123 APR18650M1A cells (counts vary by processing/paper) cycled under various fast-charging protocols; underlies the 2019 *Nature Energy* early-life-prediction paper (Severson, Attia, et al., with Braatz, Chueh, and TRI).
- A community-maintained aggregator, **`lappemic/open-source-battery-data`** on GitHub, indexes these and other open battery datasets — a useful starting point/model for the repo's own dataset index.

**Power grid**
- **IEEE distribution test feeders** (13-bus, 34-bus, 37-bus, 123-bus, 8500-node, etc.) — standard unbalanced three-phase radial-feeder benchmarks used for distribution state-estimation, OPF, and Volt/VAR-control ML research.
- **PGLib-OPF** — community-maintained library of AC-OPF test cases used as the standard benchmark suite for OPF-learning papers (should be added explicitly to the repo; not deeply covered by this search pass but well known in the field).
- **pandapower** and **PyPSA** networks — both ship with built-in synthetic/benchmark grid cases and are themselves used as synthetic-data *generators* (randomized load/generation scenarios, N-1 contingency sets) for training ML models.
- **GeoDistNet** — a newer (2026) open-source tool for generating synthetic distribution networks directly from public GIS data, exporting to pandapower format — relevant for teams needing realistic large-scale synthetic feeders beyond the classic IEEE cases.

**Nuclear**
- **CTF4Nuclear** benchmark datasets (see above) — the most directly ML-oriented open nuclear dataset/benchmark suite identified.
- OECD/NEA benchmark problem archives remain the standard reactor-physics validation datasets but are not always freely/openly downloadable; access terms vary and should be verified per-benchmark.

---

## 8. Industry Standards & Validation Context

- **ASME V&V 20-2009** — Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer; a supplement (**VVUQ 20.2**) is in development. ASME's broader **VVUQ committee structure** now explicitly includes **VVUQ 70: Verification, Validation, and Uncertainty Quantification of Artificial Intelligence and Machine Learning** — directly relevant as the standards body extending classical V&V practice to ML-based simulation surrogates.
- **Nuclear regulatory credibility framework** — NRC-oriented practice (and academic frameworks built on it) decomposes "the simulation is credible" into sub-goals spanning scenario understanding, documentation, VVUQ rigor, and quality assurance; this is the operative bar that any ML surrogate would need to clear before use in a licensing-basis nuclear safety analysis. No public evidence was found of an ML surrogate having been accepted into an NRC licensing-basis safety case as of this research pass — current ML work in nuclear T-H is predominantly for design-space exploration, non-safety-basis analysis acceleration, and research, not licensed safety calculations.
- **IEEE PES standards** — underlie state-estimation, protection, and interconnection requirements against which grid ML models are implicitly validated (e.g., IEC 60909 short-circuit standard, implemented directly in pandapower).
- **IEC 61400 series** — the governing international standard family for wind turbine design requirements (site conditions, component testing, safety, electrical integration); IEC 61400-21 extends into grid-integration/power-quality aspects relevant to storage-coupled wind systems. Small Wind Certification Council and similar bodies maintain adjacent certification standards for small wind turbines.
- **Battery safety standards**: **UL 1973** (stationary/motive battery energy storage systems, requiring full-scale thermal-runaway containment/venting tests) and **IEC 62619** (industrial secondary lithium battery safety) are the two standards most commonly cross-referenced for grid-scale battery storage certification; ML-based state-of-health/degradation models are, at present, decision-support tools layered on top of — not substitutes for — these certified safety test regimes.

**Net implication for the repo**: ML/PINN surrogates in energy are, across nuclear, grid, and battery domains alike, currently positioned as *acceleration and design-exploration tools validated against and benchmarked to* established physics codes and standards — not (yet) as replacements accepted directly into regulatory/safety-certification workflows. This is an important framing point for the repo's credibility.

---

## 9. Open Challenges & Gaps

- **Real-time control at grid scale**: RL and physics-informed control methods show strong results in simulation (voltage regulation, frequency response, grid-forming converter control), but safe deployment on live transmission/distribution systems remains limited by verifiability, N-1/N-2 contingency guarantees, and utility risk tolerance. Safe-RL formulations (Lyapunov-constrained, shielded) are an active research response but not yet standard utility practice.
- **Uncertainty quantification for safety-critical nuclear applications**: Multiple 2024–2026 papers explicitly flag that UQ for PINNs/hybrid-ML nuclear T-H models is immature relative to what licensing-grade V&V (ASME VVUQ, NRC credibility frameworks) requires; Bayesian ensembles and deep-ensemble approaches are promising but computationally heavy and not yet standardized.
- **Multi-physics coupling in batteries**: thermal-electrochemical-mechanical coupling (e.g., stress-driven degradation, thermal runaway propagation) is far less mature in the PINN/neural-operator literature than isolated electrochemical (SPM/P2D) modeling; most published PINN battery work still treats electrochemistry largely in isolation from mechanical/thermal degradation modes.
- **Data scarcity for rare grid events**: extreme events (cascading outages, large-scale blackouts, rare fault combinations) are by definition underrepresented in historical data, limiting both supervised ML and RL training; synthetic scenario generation (via pandapower/PyPSA/GeoDistNet-style tools) is a partial mitigation but synthetic-to-real generalization is unresolved.
- **Wake-model generalization**: neural operator wake models (FNO/GNO) trained on specific turbine geometries, layouts, or atmospheric stability regimes do not reliably generalize outside their training distribution; most published results (as of this search) are validated on a single reference turbine (NREL 5MW or IEA 15MW) or a narrow layout set rather than across the diversity of real wind-farm siting conditions.
- **Standardized benchmarks are new and fragmented**: unlike computer vision or NLP, physics-AI-for-energy lacks a small number of dominant, widely adopted benchmark suites. CTF4Nuclear (2026) and PGLib-OPF are steps in this direction, but wind wake modeling and battery PINNs still largely rely on ad hoc, paper-specific datasets — a clear opportunity for an open-source reference repo to fill by curating/standardizing benchmark tasks.
- **Regulatory acceptance lag**: as noted in Section 8, no clear public precedent yet exists for ML surrogates entering nuclear licensing-basis calculations or grid-code-mandated studies; closing this gap requires parallel progress on UQ, V&V documentation practice, and regulator-facing pilot studies — a multi-year effort broader than the ML research itself.
- **Vendor-reported performance claims lack independent replication**: several of the most eye-catching commercial results cited in Section 4 (10,000x, 4,000x simulation speedups) come from vendor blog posts rather than peer-reviewed, independently reproduced benchmarks — the field would benefit from open, reproducible speedup/accuracy benchmarks that are not vendor-controlled.

---

## 10. Roadmap: Building the Reference Hub

**Phase 1 (0–3 months): Foundation**
- Publish this document as the repo's `README.md`/landing reference, organized by the same taxonomy (research, architectures, industry, solvers, geometry, datasets, standards, gaps).
- Stand up a curated, tagged bibliography (BibTeX or structured YAML) covering the papers cited here plus systematic expansion per subdomain (grid, wind, battery, nuclear).
- Create a dataset index (linking OEDI, NASA/Oxford/Toyota battery datasets, IEEE test feeders, PGLib-OPF, CTF4Nuclear) with licensing notes and direct download/access instructions, modeled on existing community efforts like `lappemic/open-source-battery-data`.
- Set contribution guidelines emphasizing sourcing standards (no fabricated claims, vendor claims labeled as such, peer-reviewed vs. preprint distinction).

**Phase 2 (3–9 months): Depth and reproducibility**
- Build a small set of **reference implementations/notebooks** per domain: a PINN for the battery SPM against PyBaMM ground truth; a GNN state estimator on an IEEE test feeder via pandapower; an FNO/operator wake-model tutorial against FLORIS/AMR-Wind outputs. Reproducibility (runnable code, pinned environments) is the differentiator from purely descriptive awesome-lists.
- Start a "standards and validation" tracker mapping which ML techniques have credible V&V pathways in each domain (nuclear VVUQ 70, ASME V&V 20 supplement, IEC/UL battery standards) — kept current as standards evolve.
- Begin tracking a living leaderboard/benchmark page for the emerging standardized benchmarks (CTF4Nuclear, PGLib-OPF, and any wind-wake or battery benchmark that gains traction), explicitly separating vendor claims from independently reproduced results.

**Phase 3 (9–18 months): Community and convening**
- Seek endorsement/cross-linking from relevant open-source projects (pandapower, PyPSA, PyBaMM, FLORIS/AMR-Wind, OpenFAST) as the connective "physics AI" layer referencing their tools.
- Organize or co-organize a workshop/track at a relevant venue (PSCC, IEEE PES ML workshop series, a SciML-for-energy NeurIPS/ICML workshop) to drive contributions and visibility.
- Establish a recurring (quarterly) literature-update cadence so the repo doesn't calcify — physics-AI-for-energy is moving fast (evidenced by the volume of 2025–2026 preprints found in this research pass alone).

**Phase 4 (18–36 months): Reference-hub maturity**
- Publish an independent, reproducible benchmark report comparing physics-AI surrogates to classical solvers (pandapower/EMTP for grid, AMR-Wind/FLORIS for wind, PyBaMM/DandeLiion for battery, CTF for nuclear) on speed/accuracy/generalization — explicitly filling the "no independent replication" gap identified in Section 9.
- Formalize a "path to regulatory-grade" guidance document synthesizing ASME VVUQ 70, NRC credibility-framework literature, and IEC/UL battery/wind standards into practical guidance for practitioners trying to move ML surrogates from research to certified/licensed use.
- Aim for the repo to be the default citation/entry point that new graduate students, national-lab researchers, and industry ML teams land on when starting physics-AI-for-energy work — measured by GitHub stars/forks, citations in papers, and adoption by at least one of the open tools listed above as their recommended "ML extensions" reference.

---

## 11. Sources

Academic / research:
- [Physics-Informed Neural Networks for Power Systems (Misyris, Venzke, Chatzivasileiadis), arXiv:1911.03737](https://arxiv.org/pdf/1911.03737)
- [Physics-informed graphical neural network for power system state estimation, Applied Energy (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0306261923019669)
- [Robust Power System State Estimation using Physics-Informed Neural Networks, arXiv:2507.05874](https://arxiv.org/pdf/2507.05874)
- [Optimal Power Flow With Physics-Informed Typed Graph Neural Networks, IEEE Xplore](https://ieeexplore.ieee.org/document/10509003/)
- [Physics-informed machine learning with optimization-based guarantees: Applications to AC power flow, Applied Energy](https://www.sciencedirect.com/science/article/pii/S0142061523007986)
- [Priya L. Donti — publications](https://priyadonti.com/publications/)
- [Physics-informed Machine Learning in Power Systems, IEEE PES](https://ieee-pes.org/trending-tech/physics-informed-machine-learning-in-power-systems/)
- [A Review of Safe Reinforcement Learning Methods for Modern Power Systems, arXiv:2407.00304](https://arxiv.org/pdf/2407.00304)
- [Graph Reinforcement Learning for Power Grids: A Comprehensive Survey, arXiv:2407.04522](https://arxiv.org/pdf/2407.04522)
- [Multi-scale Dynamic Wake Modeling of Floating Offshore Wind Turbines via FNOs and PINNs, arXiv:2604.23937](https://arxiv.org/abs/2604.23937)
- [Graph Neural Operator for windfarm wake flow, Wind Energy Science preprint](https://wes.copernicus.org/preprints/wes-2025-261/)
- [Physics-informed neural network surrogate modeling of single particle model for lithium-ion batteries, arXiv:2601.01466](https://arxiv.org/pdf/2601.01466)
- [Physics-Informed Neural Network with Transfer Learning for SPMe State Estimation, arXiv:2606.28220](https://arxiv.org/pdf/2606.28220)
- [A physics-informed neural network approach to parameter estimation of lithium-ion battery electrochemical model, Journal of Power Sources](https://www.sciencedirect.com/science/article/abs/pii/S0378775324012230)
- [CTF4Nuclear: Common Task Framework for Nuclear Fission and Fusion Models, arXiv:2605.15549](https://arxiv.org/html/2605.15549v1)
- [Deployment of Traditional and Hybrid Machine Learning for CHF Prediction in CTF, arXiv:2505.14701](https://arxiv.org/pdf/2505.14701)
- [Uncertainty Quantification for Data-Driven ML Models in Nuclear Engineering, arXiv:2503.17385](https://arxiv.org/pdf/2503.17385)
- [A Deep Learning-based surrogate model for Severe Accidents using ASTEC, arXiv:2607.04450](https://arxiv.org/html/2607.04450v1)
- [Bayesian optimized deep ensemble for sodium fast reactor thermal stratification UQ, arXiv:2412.08776](https://arxiv.org/pdf/2412.08776)
- [Empirical Assessment of Time-Series Foundation Models for Power System Forecasting, arXiv:2604.22077](https://arxiv.org/html/2604.22077v1)
- [Explainable Load Forecasting with Covariate-Informed Time Series Foundation Models, arXiv:2604.28149](https://arxiv.org/html/2604.28149)
- [DandeLiion v1: An extremely fast solver for the Newman model, arXiv:2102.06534](https://arxiv.org/pdf/2102.06534)
- [Introduction to Digital Twins for the Smart Grid, arXiv:2602.14256](https://arxiv.org/pdf/2602.14256)
- [Simona Onori — Wikipedia](https://en.wikipedia.org/wiki/Simona_Onori)
- [AI accurately predicts the useful life of batteries, Stanford Energy](https://energy.stanford.edu/news/ai-accurately-predicts-useful-life-batteries-team-led-stanford-mit-finds)

Tools & infrastructure:
- [pandapower](https://www.pandapower.org/)
- [pandapower hits 500,000 downloads, Fraunhofer IEE](https://www.iee.fraunhofer.de/en/presse-infothek/press-media/2024/pandapower-open-source-tool-for-modelling-analyzing-power-grids.html)
- [GeoDistNet: Open-Source Tool for Synthetic Distribution Network Generation, arXiv:2603.29523](https://arxiv.org/pdf/2603.29523)
- [FLORIS — NREL Wind Research](https://www.nrel.gov/wind/floris)
- [FLORIS GitHub repository](https://github.com/NatLabRockies/floris)
- [SOWFA-6 GitHub / README](https://github.com/NatLabRockies/SOWFA-6/blob/master/README.SOWFA)
- [Large Eddy Simulation of wind turbine wakes, NREL](https://docs.nrel.gov/docs/fy15osti/64912.pdf)
- [IEA-15-240-RWT GitHub repository](https://github.com/IEAWindSystems/IEA-15-240-RWT)
- [EMTP — Wikipedia](https://en.wikipedia.org/wiki/EMTP)
- [DNV Bladed 5](https://www.dnv.com/services/bladed-5/)
- [Ansys Wind Turbine Design](https://ansys.synopsys.com/applications/wind-turbine-design)
- [open-source-battery-data, GitHub](https://github.com/lappemic/open-source-battery-data)

Datasets:
- [Open Energy Data Initiative (OEDI) — OpenEI](https://data.openei.org/)
- [OEDI Data Lake — AWS Open Data Registry](https://registry.opendata.aws/oedi-data-lake/)
- [Dive Into a Lake of Data: OEDI, NREL](https://www.nrel.gov/grid/news/program/2024/dive-into-a-lake-of-data-open-energy-data-initiative-increases-big-data-access-for-everyone)
- [Overview of Publicly Available Degradation Data Sets, arXiv:2403.13694](https://arxiv.org/pdf/2403.13694)

Commercial / industry:
- [Spotlight: Siemens Energy Accelerates Power Grid Asset Simulation 10,000x Using NVIDIA PhysicsNeMo](https://developer.nvidia.com/blog/spotlight-siemens-energy-accelerates-power-grid-asset-simulation-10000x-using-nvidia-physicsnemo/)
- [Near-real-time energy production forecasts with NVIDIA Earth-2 and AWS Batch](https://aws.amazon.com/blogs/hpc/aws-batch-enables-near-real-time-energy-production-forecasts-using-nvidia-earth-2/)
- [GE Vernova Digital Twin Technology](https://www.gevernova.com/software/innovation/digital-twin-technology)
- [Siemens Energy Omnivise Review, aigreentools](https://aigreentools.com/ai_tool/siemens-energy-omnivise/)
- [TerraPower and SoftServe Integrate NVIDIA Omniverse](https://www.terrapower.com/TerraPower-and-SoftServe-Integrate-NVIDIA-Omniverse-to-Speed-Advanced-Nuclear-Energy-Deployment)
- [AutoGrid — API Evangelist independent profile](https://github.com/api-evangelist/autogrid)
- [Building smart grids with Uplight's AI-driven DERMS](https://smartcar.com/blog/uplight)
- [ABB uses AI to revolutionize energy management (Verdigris)](https://new.abb.com/news/detail/41194/abb-uses-ai-to-revolutionize-energy-management)
- [Aionics targets new electrolytes for eVTOL batteries, The Air Current](https://theaircurrent.com/feed/dispatches/aionics-aviation-expansion-evtol-battery-electrolytes/)
- [Voltaiq Enterprise AI Platform](https://www.voltaiq.com/)
- [BattGenie — CB Insights profile](https://www.cbinsights.com/company/battgenie/people)

Standards:
- [ASME Verification, Validation and Uncertainty Quantification (VVUQ)](https://www.asme.org/codes-standards/publications-information/verification-validation-uncertainty)
- [A Credibility Assessment Framework Based on the Historical Use of VVUQ in Nuclear Power](https://link.springer.com/chapter/10.1007/978-3-032-09180-2_9)
- [IEC 61400 — Wikipedia](https://en.wikipedia.org/wiki/IEC_61400)
- [UL 1973 Standard for Batteries, TÜV SÜD](https://www.tuvsud.com/en-us/services/testing/energy-storage/ul-1973)
- [IEC 62619 vs. UL 1973, Kite Compliance](https://www.kitecompliance.ai/vertical-compliance/industrial-battery-standards)

*Note on sourcing rigor: this document was compiled from web/literature search in September 2026. arXiv identifiers with dates in 2026 reflect the actual search results returned and should be independently re-verified before publication, since arXiv numbering conventions and preprint dating can be easy to mis-transcribe. Vendor performance claims (speedup multipliers, ROI figures) are explicitly marked as vendor-reported and have not been independently reproduced by this research.*
