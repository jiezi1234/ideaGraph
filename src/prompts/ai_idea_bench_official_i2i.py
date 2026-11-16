from __future__ import annotations


MOTIVATION_SYSTEM_PROMPT = """\
You are an AI trained to rigorously evaluate the similarity between two research motivations by analyzing their **structural alignment**, **theoretical foundations**, and **problem focus**. Your task is to compare each pair of motivations and determine how closely they address the same core issues, challenges, or research gaps. Prioritize depth of analysis over superficial keyword matching.
"""


MOTIVATION_USER_PROMPT = """\
For each pair of motivations provided:
1. **Analyze Core Issues**: Identify the central problem(s), challenge(s), or knowledge gap(s) each motivation emphasizes.
2. **Compare Contextual Alignment**: Assess whether the motivations operate within the same domain, theoretical framework, or practical context.
3. **Evaluate Structural/Theoretical Overlap**: Determine if they share methodologies, hypotheses, or foundational theories (even if applied differently).

**Rating Criteria**:
Rate similarity on a scale of 1-5:
1. **No Similarity**: Entirely distinct problems, contexts, and methodologies.
2. **Low Similarity**: Minimal overlap in one aspect (e.g., tangential problem mention but divergent focus/theory).
3. **Moderate Similarity**: Shared problem domain but differing approaches/theories (e.g., both address climate change mitigation but focus on policy vs. technology).
4. **High Similarity**: Aligned problem focus and theory with minor differences in scope or application (e.g., both study AI bias in healthcare, but one targets diagnostics and the other patient data).
5. **Complete Similarity**: Identical problems, theories, and contextual applications.

**Output Format**
Return a JSON object for each pair with:
- A numeric `rating` (1-5)
- A concise `explanation` highlighting **specific overlaps** and **key distinctions** (1-2 sentences).

**Example Output**:
{{
  "motivation_similarity": {{
    "rating": 4,
    "explanation": "Both motivations address algorithmic bias in healthcare AI, but Motivation A focuses on diagnostic inaccuracies in radiology, while Motivation B emphasizes biases in patient prioritization systems."
  }}
}}

**Here are the motivations:**
    - Motivation 1: {motivation1}
    - Motivation 2: {motivation2}
"""


EXPERIMENT_PLAN_SYSTEM_PROMPT = """\
You are an AI trained to evaluate the similarity between two experiment plans based on their structural design, theoretical foundations, and core problem focus. Ignore superficial differences like step order, naming conventions, or formatting details.
"""


EXPERIMENT_PLAN_USER_PROMPT = """\
Compare the two provided experiment plans and assess their similarity using the criteria below.

**Evaluation Criteria:**
1. **Structural Similarity:**
   - Do the plans share a comparable framework (e.g., hypothesis testing, control groups, data analysis methods)?
   - Ignore differences in step order or terminology.

2. **Theoretical Alignment:**
   - Do they rely on the same or related theories, principles, or methodologies?

3. **Problem Focus:**
   - Do they address the same underlying problem or challenge, even if solutions differ?

**Rating Scale (1-5):**
1. **No similarity:** Entirely different problems/theories/structures.
2. **Low similarity:** Minor overlap in one criterion (e.g., same problem but divergent theories).
3. **Moderate similarity:** Align in 1-2 criteria with clear differences (e.g., shared theory but distinct structures).
4. **High similarity:** Align in 2-3 criteria with minor discrepancies (e.g., same problem and structure but different theories).
5. **Complete similarity:** Identical in all criteria.

**Output Requirements:**
- Return JSON format with `rating` (1-5) and `explanation`.
- The explanation must explicitly reference **structure**, **theory**, and **problem focus**.

**Example Output:**

{{
  "experiment_plan_similarity": {{
    "rating": 3,
    "explanation": "Both address energy efficiency (problem focus) and use quantitative metrics (structure), but Plan 1 employs machine learning (theory) while Plan 2 uses statistical modeling (theory)."
  }}
}}

**Here are the Experiment Plans:**
- Experiment Plan 1: {experiment_plan1}
- Experiment Plan 2: {experiment_plan2}
"""
