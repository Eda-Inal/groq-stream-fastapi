"""
run_multidoc_source_test.py
───────────────────────────
Runs 7 questions sequentially and records each answer as a LangSmith trace.

Documents are uploaded once at startup and deleted after all questions complete.
This works because TEST MODE is active in chat_service.py:
  - conversation_id filter is disabled → documents are found by user_id only
  - Each question uses its own conversation_id so chat history does not bleed across questions

To revert TEST MODE: see "TEST MODE" comments in chat_service.py.

Q1–Q5  — RAG questions (rag_search is triggered)
Q6–Q7  — Web questions ("internetten ara" prefix triggers web_search)

Run (Docker stack must be up):
    .venv\\Scripts\\python eval/run_multidoc_source_test.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid

import httpx

# Prevent crash when Windows terminal cannot encode UTF-8 characters
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── env ───────────────────────────────────────────────────────────────────────

def _load_env() -> None:
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

# ── config ────────────────────────────────────────────────────────────────────

BASE_URL     = "http://localhost:8000/api/v1"
HEALTH_URL   = f"{BASE_URL}/chat/models"
CHAT_URL     = f"{BASE_URL}/chat/stream"
DOCS_URL     = f"{BASE_URL}/documents"
TEST_USER_ID = "multidoc-source-test"

MODEL = "openai/gpt-oss-120b"  # Groq

PAUSE_BETWEEN_QUESTIONS = 20.0  # 3 questions/minute → 60 / 3 = 20s (RPM limit: 30)

# ── documents ─────────────────────────────────────────────────────────────────

WHITMORE_TEXT = """\
IN THE MATTER OF ESTATE OF HAROLD JAMES WHITMORE, DECEASED
Probate Case No. 2024-PR-0847 | Superior Court, County of Harwick

MEMORANDUM OF LAW IN SUPPORT OF PETITION TO CONTEST TESTAMENTARY
CAPACITY AND UNDUE INFLUENCE

Submitted by: Counsel for Petitioners Eleanor Whitmore-Callahan and Thomas Whitmore
Date: March 14, 2024

I. INTRODUCTION AND PROCEDURAL BACKGROUND

This memorandum is submitted in support of the Petition filed by Eleanor
Whitmore-Callahan and Thomas Whitmore ("Petitioners"), adult children of the decedent
Harold James Whitmore ("Decedent"), challenging the validity of the Last Will and
Testament dated September 3, 2022 ("the 2022 Will"). The 2022 Will was admitted to
probate on January 9, 2024, following Decedent's death on December 28, 2023, at age 81.

Petitioners contend that (1) the Decedent lacked testamentary capacity at the time of
execution of the 2022 Will, and (2) the 2022 Will was the product of undue influence
exercised by Respondent Diane Kowalski, Decedent's live-in caregiver from April 2021
until his death.

The 2022 Will revoked a prior will dated June 14, 2018 ("the 2018 Will"), which
distributed the estate equally among the Decedent's three children: Eleanor
Whitmore-Callahan, Thomas Whitmore, and the late Michael Whitmore (deceased June 2020).
Under the 2022 Will, the entirety of the estate — estimated at $4.2 million, comprising
a primary residence at 14 Brentwood Lane, Harwick; a vacation property in Lake Carver;
investment accounts totaling approximately $2.8 million; and miscellaneous personal
property — is bequeathed to Diane Kowalski.

II. TESTAMENTARY CAPACITY: LEGAL STANDARD

Under controlling precedent in this jurisdiction, a testator must, at the time of
execution, satisfy a four-part test: (1) know the natural objects of his bounty;
(2) understand the nature and extent of his property; (3) understand the nature of the
testamentary act itself; and (4) be capable of relating these elements to form an
orderly plan of disposition. In re Estate of Calloway, 318 Harwick App. 2d 201, 209 (2009).

Critically, the capacity required is that which existed at the moment of execution.
A testator may have lucid intervals, and a will executed during such an interval may be
valid even if the testator was otherwise incompetent. Pemberton v. Higgins, 402 Harwick
3d 87, 94 (2016).

The burden of proof rests initially with the proponent of the will to establish due
execution and that the testator was of sound mind. Once established, the burden shifts
to the contestant.

III. EVIDENCE OF LACK OF TESTAMENTARY CAPACITY

Medical records obtained from Harwick General Hospital and the Decedent's treating
physician, Dr. Susan Alvarez, reveal the following:

March 2021: Decedent diagnosed with mild cognitive impairment (MCI), with
neuropsychological testing placing him at the 14th percentile on the Montreal Cognitive
Assessment (MoCA), scoring 21 out of 30.

January 2022: Follow-up evaluation showed progression; MoCA score declined to 16 out
of 30, consistent with moderate dementia per DSM-5 criteria.

August 2022 (approximately five weeks before will execution): Dr. Alvarez's clinical
notes document that Decedent "frequently failed to recognize family members," "expressed
confusion regarding ownership of the Brentwood Lane property, believing at times it had
already been sold," and "demonstrated significant short-term memory deficits."

The drafting attorney, Mr. Gerald Finch, met with the Decedent on a single occasion on
September 3, 2022, for approximately 35 minutes. Mr. Finch did not request or review
any medical records, did not consult with Dr. Alvarez, and did not conduct or arrange
for any independent cognitive assessment. His notes reflect only that Decedent "appeared
oriented and expressed clear wishes."

Petitioners submit that Mr. Finch's cursory assessment falls below the standard of care
and is insufficient to establish testamentary capacity in a case involving documented
progressive dementia.

IV. UNDUE INFLUENCE: LEGAL STANDARD AND APPLICATION

Undue influence sufficient to void a will requires proof that: (1) the testator was
susceptible to undue influence; (2) the influencer had opportunity to exert influence;
(3) the influencer had a motive or disposition to exert influence; and (4) the will
appears to be the effect of such influence. Harwick Trust Co. v. Dellacroix, 289
Harwick 2d 344, 351 (2001).

Susceptibility: Decedent was diagnosed with progressive dementia and was wholly
dependent on Ms. Kowalski for daily living activities including bathing, medication
management, and meal preparation.

Opportunity: Ms. Kowalski resided in the Decedent's home continuously from April 2021
and was his primary and frequently sole point of contact with the outside world.
Petitioners document that Ms. Kowalski screened phone calls, managed correspondence,
and accompanied Decedent to all medical appointments after July 2022.

Motive: Ms. Kowalski had no prior familial or longstanding personal relationship with
the Decedent. She was engaged through a private caregiving agency at $28 per hour and
stands to receive the entirety of a $4.2 million estate under the contested will.

Effect: The 2022 Will represented a complete and unexplained reversal from Decedent's
longstanding testamentary intent, disinheriting his two surviving children without
stated reason.

V. RELIEF REQUESTED

Petitioners respectfully request that this Court: (1) deny probate of the 2022 Will;
(2) admit the 2018 Will to probate in its stead; and (3) award Petitioners their
reasonable attorneys' fees and costs from the estate.
"""

BRT447_TEXT = """\
JOURNAL OF BIOMEDICAL RESEARCH & INNOVATION
Vol. 12, No. 3 | September 2023 | DOI: 10.7891/JBRI.2023.0312

Efficacy of Novel mRNA-Based Therapeutic Agents in Late-Stage Pancreatic Adenocarcinoma:
A Randomized Controlled Trial
Dr. Priya Nandakumar, Dr. Stefan Holweg, Dr. Amelia Forsythe

ABSTRACT
This randomized controlled trial evaluated the efficacy and safety of BRT-447, a novel
mRNA-based therapeutic agent, in patients diagnosed with late-stage pancreatic
adenocarcinoma (Stage III-IV). A total of 284 patients were enrolled across seven
clinical sites between January 2021 and August 2022. Participants were randomized 1:1
to receive either BRT-447 (n=142) or standard-of-care gemcitabine-based chemotherapy
(n=142). The primary endpoint was overall survival (OS) at 18 months.

Results demonstrated a statistically significant improvement in OS in the BRT-447 arm:
41.3% of BRT-447 patients were alive at 18 months compared to 22.7% in the control arm
(HR=0.58, 95% CI: 0.43-0.79, p<0.001). Median PFS was 7.4 months in the BRT-447 group
versus 3.9 months in the control group. Grade 3-4 adverse events occurred in 18.3% of
BRT-447 recipients, compared to 34.6% in the control arm.

1. INTRODUCTION
Pancreatic adenocarcinoma remains one of the most lethal malignancies worldwide, with a
five-year survival rate of approximately 11% across all stages and fewer than 3% in
metastatic disease. Despite advances in surgical technique and systemic chemotherapy,
median overall survival for Stage IV disease has remained stubbornly below 12 months
for over two decades.

BRT-447 was developed by Bridgeton Therapeutics (Cambridge, MA) and encodes a modified
variant of the KRAS-G12D suppressor peptide, delivered via lipid nanoparticle (LNP)
formulation. KRAS mutations, present in over 90% of pancreatic adenocarcinomas, have
historically been considered undruggable; BRT-447 represents a novel approach to
targeting this pathway at the mRNA translation level.

2. METHODS
2.1 Study Design
Phase III, open-label, randomized controlled trial across seven sites in the United
States and Germany. Ethical approval was obtained from each site's IRB prior to
enrollment. All participants provided written informed consent.

2.2 Participants
Eligible patients were adults aged 18-75 with histologically confirmed pancreatic
adenocarcinoma (Stage III or IV), an ECOG performance status of 0-2, and no prior
exposure to mRNA-based therapeutics.

2.3 Intervention
BRT-447 was administered intravenously at 0.5 mg/kg every three weeks for up to 12
cycles. The control arm received gemcitabine 1000 mg/m2 plus nab-paclitaxel 125 mg/m2
on Days 1, 8, and 15 of each 28-day cycle, consistent with current NCCN guidelines.

3. RESULTS
3.1 Overall Survival
At the 18-month landmark, 41.3% of patients in the BRT-447 arm remained alive, compared
to 22.7% in the control arm. The hazard ratio for death was 0.58 (95% CI: 0.43-0.79),
with a log-rank p-value of <0.001.

3.2 Progression-Free Survival
Median PFS was 7.4 months (IQR: 5.1-10.2) in the BRT-447 group compared to 3.9 months
(IQR: 2.7-5.8) in the control group (HR=0.61, p<0.001).

3.3 Quality of Life
Global health status scores (EORTC QLQ-C30) at Week 12 showed a mean improvement of
14.2 points in the BRT-447 arm versus a decline of 3.1 points in the control arm.

3.4 Safety
Grade 3-4 adverse events were reported in 26 patients (18.3%) receiving BRT-447, most
commonly fatigue (6.3%), elevated liver enzymes (4.9%), and infusion-related reactions
(3.5%). In the control arm, 49 patients (34.6%) experienced Grade 3-4 adverse events,
with the most frequent being neutropenia (18.3%), nausea (9.2%), and peripheral
neuropathy (7.7%). No treatment-related deaths were reported in the BRT-447 arm.

4. DISCUSSION
The results of this trial establish BRT-447 as the first mRNA-based therapeutic to
demonstrate statistically significant improvement in overall survival in late-stage
pancreatic adenocarcinoma. The absolute improvement of 18.6 percentage points in
18-month survival is clinically substantial.

The favorable safety profile of BRT-447 relative to standard chemotherapy is
particularly noteworthy. The near-halving of Grade 3-4 adverse event rates suggests
that mRNA-based targeted therapy may circumvent the systemic toxicities that have
historically limited dose intensity in this population.

5. CONCLUSION
BRT-447 demonstrates clinically meaningful and statistically significant improvements
in overall survival, progression-free survival, and quality of life compared to standard
chemotherapy in late-stage pancreatic adenocarcinoma.

FUNDING: Bridgeton Therapeutics and the National Cancer Institute (Grant NCI-R01-CA284471).
CONFLICTS OF INTEREST: Dr. Nandakumar and Dr. Holweg report consultancy fees from
Bridgeton Therapeutics. Dr. Forsythe reports no conflicts.
"""

# ── questions ─────────────────────────────────────────────────────────────────
# "rag" questions: documents are present → rag_search is triggered automatically
# "web" questions: "internetten ara" prefix in the question forces web_search

BASE_TAGS = ["rag", "source_attribution"]

QUESTIONS = [
    {
        "id": "Q1",
        "label": "Single source — Document 1 (legal)",
        "tags": BASE_TAGS + ["single_source"],
        "question": (
            "What was Harold Whitmore's MoCA score in January 2022, "
            "and what did it indicate about his cognitive condition?"
        ),
    },
    {
        "id": "Q2",
        "label": "Single source — Document 2 (academic)",
        "tags": BASE_TAGS + ["single_source"],
        "question": (
            "What percentage of BRT-447 patients experienced Grade 3–4 adverse events, "
            "and what were the most common ones?"
        ),
    },
    {
        "id": "Q3",
        "label": "Cross-document — cognitive measurement tools",
        "tags": BASE_TAGS + ["cross_document"],
        "question": (
            "One document discusses a person's mental deterioration, and another discusses "
            "cognitive assessment tools used in clinical trials. What do both documents "
            "reveal about how cognitive decline is measured?"
        ),
    },
    {
        "id": "Q4",
        "label": "Cross-document — financial motivation",
        "tags": BASE_TAGS + ["cross_document"],
        "question": (
            "Compare the role of financial motivation in both documents — "
            "how does money influence the behavior of key individuals in each case?"
        ),
    },
    {
        "id": "Q5",
        "label": "Cross-document — vulnerability",
        "tags": BASE_TAGS + ["cross_document"],
        "question": (
            "Both documents involve vulnerable individuals. How is vulnerability "
            "defined or described differently in each document?"
        ),
    },
    {
        "id": "Q6",
        "label": "Web search — FDA approval status of mRNA therapeutics",
        "tags": BASE_TAGS + ["web_fallback"],
        "question": (
            "internetten ara: What is the current FDA approval status of mRNA-based "
            "therapeutics for pancreatic cancer as of 2024?"
        ),
    },
    {
        "id": "Q7",
        "label": "Web search — US legal standards for contesting a will",
        "tags": BASE_TAGS + ["web_fallback"],
        "question": (
            "internetten ara: What are the general legal standards for contesting a will "
            "on grounds of undue influence in the United States today?"
        ),
    },
]

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _upload_doc(filename: str, text: str, conversation_id: str | None = None) -> int:
    payload: dict = {
        "filename": filename,
        "text": text,
        "document_type": "text",
        "user_id": TEST_USER_ID,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    r = httpx.post(DOCS_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["document_id"]


def _delete_doc(doc_id: int) -> None:
    try:
        httpx.delete(f"{DOCS_URL}/{doc_id}", timeout=10)
    except Exception:
        pass


def _chat(question: str, conversation_id: str, tags: list[str] | None = None) -> str:
    payload: dict = {
        "messages": [{"role": "user", "content": question}],
        "user_id": TEST_USER_ID,
        "conversation_id": conversation_id,
        "temperature": 0,
        "tags": tags or [],
    }
    if MODEL:
        payload["model"] = MODEL
    full_text = ""
    with httpx.stream("POST", CHAT_URL, json=payload, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                full_text += delta.get("content", "")
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
    return full_text


def _warmup(retries: int = 3, delay: float = 3.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            print(f"API health check (attempt {attempt}/{retries})...", end=" ", flush=True)
            httpx.get(HEALTH_URL, timeout=10).raise_for_status()
            print("OK")
            return
        except Exception as e:
            print(f"failed: {e}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError("API health check failed. Is the Docker stack running?")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _warmup()
    print()
    print("=" * 70)
    print(f"  Running 7 questions  |  user_id={TEST_USER_ID}")
    print(f"  Documents uploaded once, deleted after all questions complete.")
    print(f"  Each question appears as a separate trace in LangSmith.")
    print("=" * 70)

    # Upload documents once — no conversation_id, found by user_id in TEST MODE
    uploaded_ids: list[int] = []
    try:
        doc_id_1 = _upload_doc("whitmore_estate.txt", WHITMORE_TEXT, conversation_id=None)
        uploaded_ids.append(doc_id_1)
        doc_id_2 = _upload_doc("brt447_trial.txt", BRT447_TEXT, conversation_id=None)
        uploaded_ids.append(doc_id_2)
        print(f"\n  Documents uploaded: whitmore_estate.txt (id={doc_id_1}), brt447_trial.txt (id={doc_id_2})")
    except Exception as e:
        print(f"  ERROR — document upload failed: {e}")
        return

    print()

    try:
        for i, q in enumerate(QUESTIONS, 1):
            conversation_id = str(uuid.uuid4())

            print(f"[{i}/7] {q['id']} — {q['label']}")
            print(f"  conv_id  : {conversation_id}")
            print(f"  tags     : {q['tags']}")
            print(f"  question : {q['question'][:80]}...")

            try:
                answer = _chat(q["question"], conversation_id, tags=q["tags"])
                print()
                print("  ── ANSWER " + "─" * 57)
                for line in answer.strip().splitlines():
                    print(f"  {line}")
                print("  " + "─" * 66)
            except Exception as e:
                print(f"  ERROR: {e}")

            if i < len(QUESTIONS):
                time.sleep(PAUSE_BETWEEN_QUESTIONS)
            print()

    finally:
        for doc_id in uploaded_ids:
            _delete_doc(doc_id)
        print(f"  Documents deleted: {uploaded_ids}")

    print("=" * 70)
    print("  Done.")
    print("  LangSmith traces: https://smith.langchain.com > Projects > groq-stream-fastapi")
    print("=" * 70)


if __name__ == "__main__":
    main()
