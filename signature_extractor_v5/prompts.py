# signature_extractor_v5/prompts.py
"""Prompt templates for signature extraction."""

SIGNATURE_DETECTION_PROMPT = """Analyze this document page for signatures.

Your task:
1. Detect if any signatures are present (handwritten, typed, or digital)
2. Count the number of signatures
3. Identify signature characteristics

Look for:
- Handwritten signatures or initials
- Signature blocks with "Signed:", "Signature:", etc.
- Names with titles and dates in signature contexts
- Signature lines or designated signature areas

Output in this exact format:
HAS_SIGNATURES: [yes/no]
SIGNATURE_COUNT: [number]
SIGNATURE_TYPE: [handwritten/typed/digital/mixed]
IS_IMAGE: [yes/no]
IS_SCANNED: [yes/no]
CONFIDENCE: [high/medium/low]

Be concise and precise."""


SIGNATURE_EXTRACTION_PROMPT = """Extract signature information from this document page.

Your task:
1. Extract all signatory names and titles
2. Extract associated dates
3. Extract company/organization information

For each signature found, extract:
- Full name of signatory
- Job title or position
- Company/organization name
- Date of signature

Output in this exact format:
SIGNATORY_1:
NAME: [full name]
TITLE: [job title]
COMPANY: [organization]
DATE: [date if present]

SIGNATORY_2:
[same format]

If no signatures found, output:
NO_SIGNATURES_FOUND

Be precise and extract only what is clearly visible."""