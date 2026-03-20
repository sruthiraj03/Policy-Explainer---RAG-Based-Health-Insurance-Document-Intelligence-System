# User Guide

This guide explains how to use the PolicyExplainer application.

PolicyExplainer helps you understand complex health insurance policy documents by generating structured summaries, grounded answers, FAQs, and evaluation-backed reliability metrics.

No technical background is required.

---

# Application Overview

PolicyExplainer allows you to:

- Upload a health insurance policy PDF
- Generate structured, section-wise summaries
- Ask grounded questions using a document-aware assistant
- View policy-specific FAQs
- Download a structured summary
- See confidence indicators for outputs
- Evaluate summary quality using measurable metrics

All outputs are strictly grounded in your uploaded document.

If information is not found, the system will respond:

```text
Not found in this document.
```

No external knowledge is used.

---

# Step 1: Upload Your Policy

When you open the application, you will see:

- A header describing PolicyExplainer
- A file upload section
- A "Browse files" button

To begin:

1. Click "Browse files"
2. Select your insurance policy PDF
3. Wait for processing to complete

Once ingestion finishes, you will be redirected to the main dashboard.

---

# Main Dashboard

After uploading a policy, the dashboard includes:

- Top navigation:
  - Summary
  - FAQs
  - Save Insurance Summary
  - New Policy
- Policy Highlights (left or center)
- Policy Assistant chat interface (right side)

---

# Summary (Policy Highlights)

The Summary tab shows structured highlights of your policy.

Typical sections include:

- Plan Snapshot
- Cost Summary
- Covered Services
- Administrative Conditions
- Exclusions and Limitations
- Claims and Appeals

Each section:

- Is collapsible
- Contains bullet-point summaries
- Includes page-level citations
- Displays a confidence level

Example:

```text
Confidence: HIGH • Annual deductible is $1500 (p. 3)
```

Confidence reflects how strongly the content is supported by the document.

---

# Policy Assistant (Ask Questions)

The Policy Assistant allows you to ask natural language questions about your policy.

Example questions:

- What is my deductible?
- Is emergency care covered?
- Do I need prior authorization?
- What services are excluded?

How to use:

1. Enter your question in the input box
2. Submit your query
3. Review the response

The system will:

- Retrieve relevant document sections
- Generate a grounded answer
- Validate citations
- Display a confidence indicator

If no supporting information is found:

```text
Not found in this document.
```

---

# FAQs (Frequently Asked Questions)

The FAQs tab generates policy-specific questions and answers.

Features:

- Automatically generated from your document
- Focused on important coverage areas
- Grounded in policy content
- Structured for easy reading
- Supported by citations

Use FAQs to quickly understand key aspects of your policy.

---

# Save Insurance Summary (Download)

You can download a structured summary of your policy.

Steps:

1. Click "Save Insurance Summary"
2. A formatted summary file is generated
3. Save it locally

The downloaded summary includes:

- Section-wise breakdown
- Bullet-point highlights
- Page references

Useful for sharing or offline review.

---

# New Policy

To analyze another policy:

1. Click "New Policy"
2. Upload a new PDF
3. Previous document context is cleared
4. A new document session is created

Each policy is processed independently.

---

# Confidence Levels

Each generated output includes a confidence level.

Confidence is based on:

- Citation validity
- Number of supporting citations
- Retrieval strength
- Validation checks

It indicates how well the output is supported by the document, not legal accuracy.

---

# Evaluation Metrics

PolicyExplainer evaluates summary quality using three metrics.

## 1. Faithfulness

Measures whether each summary statement is supported by cited text.

Higher score = stronger grounding.

---

## 2. Completeness

Measures how well the summary covers key policy sections.

Higher score = more comprehensive coverage.

---

## 3. Simplicity

Measures how easy the summary is to understand compared to the original policy.

Considers:

- Sentence simplicity
- Reduced jargon
- Improved readability
- Clear structure

Higher score = easier to understand.

---

# What Happens Behind the Scenes

**When you upload a policy:**

- The document is parsed and cleaned
- Text is split into chunks
- Embeddings are generated for retrieval

**When generating outputs:**

- Relevant sections are retrieved
- The model generates structured responses
- Citations are validated
- Unsupported content is removed

**When evaluating:**

- Outputs are compared against source text
- Scores are computed deterministically

---

# Important Limitations

PolicyExplainer:

- Does not provide medical advice
- Does not provide legal advice
- Does not interpret intent beyond document text
- Cannot process scanned PDFs without OCR

If your PDF is image-based, text extraction may fail.

---

# Best Practices

For best results:

- Ask specific and clear questions
- Review citations for important decisions
- Start with the summary before asking questions
- Use FAQs for quick understanding
- Download summaries for reference

---

# Data Handling

- Uploaded documents are stored locally
- Documents are not used for model training
- API keys are stored securely in environment variables
- Data retention depends on deployment setup

---

# Summary

PolicyExplainer is designed to make health insurance policies:

- Clear
- Structured
- Grounded
- Measurable

It prioritizes reliability and transparency over speculation.

---

*End of User Guide.*
