# Prompt Engineering for IPM Derivation Competition

This repository contains the public materials for the **Prompt Engineering for IPM Derivation Competition**, a proposed challenge for the ICST 2027 Challenge Competition Track.

The goal of the competition is to evaluate how well prompts can guide Large Language Models (LLMs) in deriving **Input Parameter Models (IPMs)** from textual specifications of **Systems Under Test (SUTs)**:
- Participants submit their prompts 
- The competition infrastructure runs those prompts on benchmark specifications and evaluates the generated IPMs against reference models

![Competition workflow](assets/Workflow.png)

---

## Motivation

Combinatorial Testing (CT) relies on the construction of Input Parameter Models. An IPM describes the parameters of a system, their possible values, and the constraints that restrict valid combinations. Building such models is often manual and can be difficult for newcomers to CT.

This competition uses LLM-assisted IPM derivation as a low-barrier entry point into CT. Instead of implementing a new covering-array generator, participants work on the prompt-engineering problem of helping an LLM extract a useful IPM from a textual SUT description. This connects CT with the active area of LLM-assisted software testing and produces reusable artifacts for future research.

The competition also aims to answer the following research questions:

1. **RQ1:** To what extent can prompt-engineered LLMs derive syntactically valid and semantically accurate IPMs from textual SUT specifications?
2. **RQ2:** Which parts of IPM derivation are most challenging for LLMs: parameter identification, value extraction, or constraint derivation?
3. **RQ3:** Do prompts optimized on a known LLM generalize to unseen SUT specifications and to different LLMs?

---

## Important dates

The following dates are tentative and will be confirmed on the official ICST page, if the competition is accepted.

| Activity | Date |
| --- | --- |
| Competition website and repository available | 31 October 2026 |
| Competition opens | 1 December 2026 |
| Participant submission deadline | 15 January 2027 |
| Notification of offline evaluation results | 15 February 2027 |
| Competition report submission | 28 February 2027 |

---

## Who can participate

The competition is open to researchers, students, practitioners, and teams interested in software testing, combinatorial testing, prompt engineering, and LLM-assisted testing.

Participants do **not** need to be experts in combinatorial testing. The development benchmarks, reference IPMs, scripts, and examples are intended to make the task accessible while still enabling rigorous comparison of submitted approaches.

---

## Competition task

Given a textual specification of a SUT, participants must design a prompt that instructs an LLM to generate a corresponding IPM in the **ACTS format**.

A generated IPM should correctly identify:

- the system parameters;
- the type of each parameter;
- the possible values of each parameter;
- the constraints among parameters and values.

### Example input specification

```text
A browser configuration system supports three operating systems: Windows, Linux, and macOS.
The browser can be Edge, Firefox, or Safari. Safari is not available on Windows or Linux.
```

### Example expected ACTS-style output

```text
[System]
Name: BrowserConfiguration

[Parameter]
OS (enum): Windows, Linux, macOS
Browser (enum): Edge, Firefox, Safari

[Constraint]
OS = "Windows" => Browser != "Safari"
OS = "Linux" => Browser != "Safari"
```

The example above is illustrative. Competition benchmarks may contain different domains, parameter types, and constraints.

---

## What participants submit

Participants will submit:

1. a **nickname** (or team name), used for the public ranking, which will be posted on this repository;
2. a **prompt template** containing the literal placeholder `{SPECIFICATION}`;
3. optionally, a short participant paper describing the prompt-engineering strategy, experimental evaluation, and lessons learned.

A valid prompt template must contain `{SPECIFICATION}` exactly once or more. During execution, our runner replaces this placeholder with the textual specification of each benchmark SUT.

### Minimal prompt template example

```text
You are given the textual specification of a system under test.
Generate an Input Parameter Model in ACTS format.
Return only the ACTS model. Do not include Markdown fences or explanations.

Specification:
{SPECIFICATION}
```