# Accelerating scikit-learn on Intel hardware: Our first progress update

By Yann Lechelle, Executive President & Chairman of Probabl

TL;DR: A six-month check-in on the Probabl-Intel collaboration on scikit-learn. What we set out to do, what we've delivered so far, and where we're heading next.

## Where we started

*"Enterprises run scikit-learn at scale while performance and cost directly shape what they can build. Our work with Intel is about delivering measurable gains in both, openly and for everyone, so the whole community moves forward together."* – Yann Lechelle, Executive President, Probabl

In June, at VivaTech in Paris, Probabl and Intel announced a collaboration to optimize and advance scikit-learn so that data scientists and enterprises get the best possible performance and total cost of ownership from Intel hardware.

The collaboration brings together Probabl, the official steward of scikit-learn, and Intel. We’re working hand in hand to make scikit-learn run faster and more efficiently on Intel platforms,  including but not limited to Intel Xeon CPUs and Intel GPUs, with a particular focus on the data center environments where performance and cost-efficiency at scale matter most.

From the outset, the initiative has been built on a shared commitment to open source. Improvements developed through this work are contributed back to the scikit-learn project under its permissive BSD license, benefiting the entire community rather than any single vendor. The goal is straightforward: better out-of-the-box performance for the millions of users who rely on scikit-learn every day, and a clear path for organizations to maximize the value of their Intel-based infrastructure for tabular AI and machine learning.

This post is our first progress update since that announcement. I share a recap of the technical work delivered so far and the priorities that will shape the next phase.

## What we've delivered so far

[Labs: this is the core of the update. Please describe the work shipped since June, e.g. what changed, why it matters, and where possible measurable results. Link to relevant PRs, issues, and release notes.]

[My answer: We’ve mostly delivered benchmarks, issues, upstream reports, unmerged PRs, benchmarks helped detect performance problems and bugs in scikit-learn and in upstream libraries.]

## Contributing back to the community

[My comment: "Contributing back" why back? It's not back, it's from the start that we contribute to the community]

Everything above lands where it belongs: in scikit-learn itself, but also in the scientific Python ecosystem (including pytorch, openBLAS, oneDAL). By aligning Probabl's core expertise with Intel's leadership in compute hardware, this joint effort aims to set a new standard for performance and efficiency in everyday machine learning from individual practitioners to large-scale data center deployments.
Because the improvements ship upstream in scikit-learn, users see the benefit without changing how they work: the same familiar API, running faster on the hardware they already have.

## What's next: upcoming priorities

[Labs: outline the roadmap for the next phase, e.g. 3 priorities.]

Priorities are:
- Improve performance documentation in scikit-learn to relate findings from the benchmarks
- Fix HGB scalability issues
- Improve encoders (found to be an easy-to-fix bottleneck in some pipelines)
- Rework trees implementation to introduce binning, which will speed it up considerably (RandomForests mainly)
- Extend benchmarks
