# Content and design evaluation

## Recommended public content

The homepage should answer four questions quickly:

1. Who is Ray? An NYU computer science and mathematics undergraduate.
2. What is the research focus? Learned robot collectives, decentralized control, and interpretable collective intelligence.
3. What evidence supports that focus? Current swarm work at NYU, representation learning at Stanford, interaction analysis at the Genetic Heritage Group, and resource-aware optimization at Galatea Bio.
4. How can a visitor follow up? Email, LinkedIn, and a downloadable CV.

The strongest narrative is not a chronological résumé. It is a research arc:

- **Local representations:** geometry-aware and set-level encodings for limited observations.
- **Distributed learning:** local policies that produce coherent global morphology.
- **Scalable training:** resource-aware simulation and optimization.
- **Interpretability:** methods that connect interactions to collective outcomes.

This arc comes directly from the statement of purpose and makes the site useful to faculty, research collaborators, and graduate-admissions readers.

## What was intentionally excluded

- **Phone number:** unnecessary for a public research site and better kept on the CV.
- **Full coursework and every résumé bullet:** the website should scan quickly; the CV holds the complete record.
- **The statement of purpose PDF:** it is application-specific and contains draft annotations, so it should not be published.
- **GitHub, Google Scholar, paper, and project links:** these were not present in the supplied documents. Add them when accurate URLs are available rather than publishing placeholders.
- **Personal interests:** none were supplied, so no hobbies or personality details were invented.

## Design rationale

The design combines three useful patterns from the reference sites:

- A compact identity rail and straightforward research hierarchy.
- A conversational, text-first introduction with a restrained set of links.
- Visually distinct research entries with concise metadata and summaries.

The result remains lightweight: plain HTML, CSS, and JavaScript with no build system, analytics, cookies, or external font dependency. The portrait area can optionally run a Cells2Pixels-compatible NCA in WebGL2; phones, reduced-motion users, slow loads, and unsupported browsers receive the static portrait instead. The generated canvas preserves learned alpha transparency so it blends into the page rather than rendering on a white rectangle.

## Suggested additions later

Add only when the material is public and polished:

- A professional background-removed RGBA portrait plus a Cells2Pixels-trained growing checkpoint for the live NCA portrait.
- GitHub and Google Scholar links.
- Paper URLs, code repositories, videos, and project pages.
- One real figure or animation for each major research project.
- Awards, teaching, invited talks, or news if these become substantial enough to merit a compact section.
