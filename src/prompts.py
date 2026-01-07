"""
Prompt templates for syllogistic reasoning.
"""
from typing import Callable


def direct_prompt(syllogism: str) -> str:
    """
    Simple direct prompt asking for validity judgment.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: Evaluate FORMAL LOGICAL VALIDITY only. Completely ignore whether the argument sounds plausible or realistic.

KEY PRINCIPLE:
- VALID = conclusion MUST be true IF premises are true (regardless of content)
- INVALID = conclusion does NOT necessarily follow from premises (even if plausible)

WARNING - Content Bias:
- Plausible arguments can be INVALID: "Most doctors are smart. John is smart. Therefore John is a doctor."
- Absurd arguments can be VALID: "All cats are reptiles. All reptiles are purple. Therefore all cats are purple."

IGNORE: Real-world truth, plausibility, whether it "makes sense"
EVALUATE: Does conclusion follow by logical necessity from premises?

Argument:
{syllogism}

Analyze ONLY logical structure, not content. Assume you know nothing about the real world.

Respond with JSON:
{{
    "validity": true or false
}}"""

    return prompt


def chain_of_thought_prompt(syllogism: str) -> str:
    """
    Chain-of-thought prompt with step-by-step reasoning.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: Evaluate FORMAL LOGICAL VALIDITY only. Ignore plausibility and real-world knowledge completely.

KEY: Absurd content can be VALID. Realistic content can be INVALID. Judge only logical structure.

Argument:
{syllogism}

Follow these steps:

1. IDENTIFY: What are the premises and conclusion? (Ignore plausibility)

2. CONVERT to form: "All/No/Some X are Y"
   - Identify: major term (conclusion predicate), minor term (conclusion subject), middle term (in premises only)
   - Don't let content influence this step

3. CHECK structure:
   - Is middle term distributed in at least one premise?
   - Watch for fallacies: undistributed middle, affirming consequent, denying antecedent, illicit major/minor

4. EVALUATE: If premises are true (even if absurd), MUST conclusion be true?
   - Valid argument can have false premises/conclusion
   - Invalid argument can have true premises/conclusion

5. DECIDE: VALID = conclusion follows necessarily (regardless of content)

IGNORE: Real-world truth, plausibility, whether it "makes sense"

Respond with JSON:
{{
    "validity": true or false
}}"""

    return prompt


def normalization_prompt(syllogism: str) -> str:
    """
    Prompt for rewriting the three sentences of a syllogism
    into canonical categorical form.
    """

    prompt = f"""You are a logic expert. Your task is to normalize a syllogistic argument into canonical form.

INPUT STRUCTURE:
You will receive exactly THREE sentences in this order:
1. First sentence = MAJOR PREMISE
2. Second sentence = MINOR PREMISE  
3. Third sentence = CONCLUSION

CRITICAL TASK:
Map each sentence to EXACTLY one of these four canonical forms:
- "All X are Y."
- "No X are Y."
- "Some X are Y."
- "Some X are not Y."

CRUCIAL RULES:
1. PRESERVE the exact order: major premise, minor premise, conclusion
2. Do NOT reorder or swap sentences
3. Do NOT add, remove, or infer information
4. Do NOT change the logical meaning
5. ONLY change the wording to match one of the four canonical forms
6. Keep the same terms - do NOT introduce synonyms
7. Each sentence MUST end with a period

QUANTIFIER MAPPING EXAMPLES:
- "All", "every", "any", "each" → "All"
- "No", "none", "not any" → "No"  
- "Some", "a few", "many", "most", "there are", "there exist" → "Some"
- "Some...not", "not all", "not every" → "Some...not"


SYLLOGISM TO NORMALIZE:
{syllogism}

OUTPUT FORMAT:
Respond with ONLY the three normalized sentences, separated by spaces. No JSON, no markdown, no explanation, no numbering.

Format: [major premise]. [minor premise]. [conclusion]."""

    return prompt

def normalization_replace_prompt(syllogism: str) -> str:
    """
    Normalizes the syllogism AND replaces all concrete terms
    with abstract variables A, B, C (canonical form).
    """

    prompt = f"""You are a logic expert. Your task is to normalize a syllogistic argument into PURELY FORMAL canonical form.

INPUT STRUCTURE:
You will receive exactly THREE sentences in this order:
1. MAJOR PREMISE
2. MINOR PREMISE
3. CONCLUSION

STEP 1 — NORMALIZATION:
Rewrite each sentence into EXACTLY one of:
- "All X are Y."
- "No X are Y."
- "Some X are Y."
- "Some X are not Y."

STEP 2 — VARIABLE REPLACEMENT:
Replace ALL concrete terms with abstract variables:
- Use single capital letters: A, B, C
- Use the MINIMUM number of variables
- The SAME original term must map to the SAME variable everywhere
- Different terms must map to DIFFERENT variables

EXAMPLE:
Input:
All cats are mammals. All mammals are animals. Therefore all cats are animals.

Output:
All A are B. All B are C. All A are C.

RULES:
- Preserve sentence order
- Do NOT reorder premises
- Do NOT add or remove information
- Do NOT explain anything
- Each sentence MUST end with a period

SYLLOGISM:
{syllogism}

OUTPUT FORMAT:
Exactly three sentences, separated by spaces.
No JSON. No markdown. No explanations.
"""

    return prompt


# Dictionary of available prompt templates
PROMPT_TEMPLATES: dict[str, Callable[[str], str]] = {
    "direct": direct_prompt,
    "chain_of_thought": chain_of_thought_prompt,
    "cot": chain_of_thought_prompt,
    "normalization": normalization_prompt,
    "normalize": normalization_prompt,
    "normalization_replace": normalization_replace_prompt,
    "normalize_replace": normalization_replace_prompt,
}


def get_prompt_template(template_name: str) -> Callable[[str], str]:
    """
    Get a prompt template by name.

    Args:
        template_name: Name of the template ('direct' or 'chain_of_thought'/'cot')

    Returns:
        Prompt template function

    Raises:
        ValueError: If template name is not recognized
    """
    if template_name not in PROMPT_TEMPLATES:
        available = ", ".join(PROMPT_TEMPLATES.keys())
        raise ValueError(
            f"Unknown prompt template: {template_name}. "
            f"Available templates: {available}"
        )

    return PROMPT_TEMPLATES[template_name]
