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


# Dictionary of available prompt templates
PROMPT_TEMPLATES: dict[str, Callable[[str], str]] = {
    "direct": direct_prompt,
    "chain_of_thought": chain_of_thought_prompt,
    "cot": chain_of_thought_prompt,  # alias
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
