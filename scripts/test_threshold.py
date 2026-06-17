"""
Rerank threshold analysis test.

Uploads a legal document and sends 20 questions (FOUND / PARTIAL / NONE).
Check LangSmith for rerank scores after each question.

Model: openai/gpt-oss-120b:free
Rate limits: 30 RPM | 1K RPD | 8K TPM | 200K TPD
Delay: 30s between questions (safe for 8K TPM)

Run: .venv\Scripts\python scripts\test_threshold.py
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
from uuid import uuid4

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000/api/v1"
MODEL = "llama-3.1-8b-instant"
USER_ID = "test_user"
CONVERSATION_ID = "3.70-test-threshold-0c1b6ef9"  # reuse document from 3.70b test
FILENAME = "whitmore_estate_78076a.md"             # same document, no re-upload
SKIP_UPLOAD = True
DELAY_SECONDS = 40

DOCUMENT_TEXT = """IN THE MATTER OF ESTATE OF HAROLD JAMES WHITMORE, DECEASED
Probate Case No. 2024-PR-0847 | Superior Court, County of Harwick

MEMORANDUM OF LAW IN SUPPORT OF PETITION TO CONTEST TESTAMENTARY CAPACITY AND UNDUE INFLUENCE

Submitted by: Counsel for Petitioners Eleanor Whitmore-Callahan and Thomas Whitmore
Date: March 14, 2024

I. INTRODUCTION AND PROCEDURAL BACKGROUND

This memorandum is submitted in support of the Petition filed by Eleanor Whitmore-Callahan and Thomas Whitmore ("Petitioners"), adult children of the decedent Harold James Whitmore ("Decedent"), challenging the validity of the Last Will and Testament dated September 3, 2022 ("the 2022 Will"). The 2022 Will was admitted to probate on January 9, 2024, following Decedent's death on December 28, 2023, at age 81.

Petitioners contend that (1) the Decedent lacked testamentary capacity at the time of execution of the 2022 Will, and (2) the 2022 Will was the product of undue influence exercised by Respondent Diane Kowalski, Decedent's live-in caregiver from April 2021 until his death.

The 2022 Will revoked a prior will dated June 14, 2018 ("the 2018 Will"), which distributed the estate equally among the Decedent's three children: Eleanor Whitmore-Callahan, Thomas Whitmore, and the late Michael Whitmore (deceased June 2020). Under the 2022 Will, the entirety of the estate — estimated at $4.2 million, comprising a primary residence at 14 Brentwood Lane, Harwick; a vacation property in Lake Carver; investment accounts totaling approximately $2.8 million; and miscellaneous personal property — is bequeathed to Diane Kowalski.

II. TESTAMENTARY CAPACITY: LEGAL STANDARD

Under controlling precedent in this jurisdiction, a testator must, at the time of execution, satisfy a four-part test: (1) know the natural objects of his bounty; (2) understand the nature and extent of his property; (3) understand the nature of the testamentary act itself; and (4) be capable of relating these elements to form an orderly plan of disposition. In re Estate of Calloway, 318 Harwick App. 2d 201, 209 (2009).

Critically, the capacity required is that which existed at the moment of execution. A testator may have lucid intervals, and a will executed during such an interval may be valid even if the testator was otherwise incompetent. Pemberton v. Higgins, 402 Harwick 3d 87, 94 (2016).

The burden of proof rests initially with the proponent of the will to establish due execution and that the testator was of sound mind. Once established, the burden shifts to the contestant.

III. EVIDENCE OF LACK OF TESTAMENTARY CAPACITY

Medical records obtained from Harwick General Hospital and the Decedent's treating physician, Dr. Susan Alvarez, reveal the following:

March 2021: Decedent diagnosed with mild cognitive impairment (MCI), with neuropsychological testing placing him at the 14th percentile on the Montreal Cognitive Assessment (MoCA), scoring 21 out of 30.

January 2022: Follow-up evaluation showed progression; MoCA score declined to 16 out of 30, consistent with moderate dementia per DSM-5 criteria.

August 2022 (approximately five weeks before will execution): Dr. Alvarez's clinical notes document that Decedent "frequently failed to recognize family members," "expressed confusion regarding ownership of the Brentwood Lane property, believing at times it had already been sold," and "demonstrated significant short-term memory deficits."

The drafting attorney, Mr. Gerald Finch, met with the Decedent on a single occasion on September 3, 2022, for approximately 35 minutes. Mr. Finch did not request or review any medical records, did not consult with Dr. Alvarez, and did not conduct or arrange for any independent cognitive assessment. His notes reflect only that Decedent "appeared oriented and expressed clear wishes."

Petitioners submit that Mr. Finch's cursory assessment falls below the standard of care and is insufficient to establish testamentary capacity in a case involving documented progressive dementia.

IV. UNDUE INFLUENCE: LEGAL STANDARD AND APPLICATION

Undue influence sufficient to void a will requires proof that: (1) the testator was susceptible to undue influence; (2) the influencer had opportunity to exert influence; (3) the influencer had a motive or disposition to exert influence; and (4) the will appears to be the effect of such influence. Harwick Trust Co. v. Dellacroix, 289 Harwick 2d 344, 351 (2001).

Courts have recognized that susceptibility is heightened where the testator suffers from cognitive decline, social isolation, or dependency on the alleged influencer.

Susceptibility: Decedent was diagnosed with progressive dementia and was wholly dependent on Ms. Kowalski for daily living activities including bathing, medication management, and meal preparation.

Opportunity: Ms. Kowalski resided in the Decedent's home continuously from April 2021 and was his primary and frequently sole point of contact with the outside world. Petitioners document that Ms. Kowalski screened phone calls, managed correspondence, and accompanied Decedent to all medical appointments after July 2022.

Motive: Ms. Kowalski had no prior familial or longstanding personal relationship with the Decedent. She was engaged through a private caregiving agency at $28 per hour and stands to receive the entirety of a $4.2 million estate under the contested will.

Effect: The 2022 Will represented a complete and unexplained reversal from Decedent's longstanding testamentary intent, disinheriting his two surviving children without stated reason.

V. RELIEF REQUESTED

Petitioners respectfully request that this Court: (1) deny probate of the 2022 Will; (2) admit the 2018 Will to probate in its stead; and (3) award Petitioners their reasonable attorneys' fees and costs from the estate.
"""

QUESTIONS = [
    {"id": "Q1",  "label": "FOUND",   "text": "What was Harold James Whitmore's MoCA score in January 2022?"},
    {"id": "Q2",  "label": "FOUND",   "text": "On what date was the contested 2022 Will executed?"},
    {"id": "Q3",  "label": "FOUND",   "text": "How long did attorney Gerald Finch's meeting with the Decedent last on the day of execution?"},
    {"id": "Q4",  "label": "FOUND",   "text": "What was the hourly rate paid to Diane Kowalski through the caregiving agency?"},
    {"id": "Q5",  "label": "FOUND",   "text": "What is the total estimated value of the estate under the 2022 Will?"},
    {"id": "Q6",  "label": "FOUND",   "text": "What MoCA score did the Decedent receive in March 2021, and what percentile did that represent?"},
    {"id": "Q7",  "label": "FOUND",   "text": "What specific observations did Dr. Alvarez record in her August 2022 clinical notes regarding the Decedent's condition?"},
    {"id": "Q8",  "label": "PARTIAL", "text": "Would the 2022 Will be invalidated if the Decedent had a lucid interval at the time of signing?"},
    {"id": "Q9",  "label": "PARTIAL", "text": "What standard of care should a drafting attorney follow when a testator has documented progressive dementia?"},
    {"id": "Q10", "label": "PARTIAL", "text": "Did Diane Kowalski have any familial relationship with the Decedent prior to becoming his caregiver?"},
    {"id": "Q11", "label": "PARTIAL", "text": "How did the distribution of the estate differ between the 2018 Will and the 2022 Will?"},
    {"id": "Q12", "label": "PARTIAL", "text": "Under what legal test is undue influence established in this jurisdiction?"},
    {"id": "Q13", "label": "PARTIAL", "text": "What medical records did Gerald Finch review before drafting the 2022 Will?"},
    {"id": "Q14", "label": "PARTIAL", "text": "When did Michael Whitmore die, and how does his death affect the 2018 Will's distribution?"},
    {"id": "Q15", "label": "NONE",    "text": "What were the exact investment account balances as of the date of death?"},
    {"id": "Q16", "label": "NONE",    "text": "Did Dr. Alvarez testify in any prior court proceeding involving the Whitmore family?"},
    {"id": "Q17", "label": "NONE",    "text": "What is the statute of limitations for contesting a will in Harwick jurisdiction?"},
    {"id": "Q18", "label": "NONE",    "text": "Has Diane Kowalski faced any prior allegations of elder financial abuse or undue influence in previous caregiving roles?"},
    {"id": "Q19", "label": "NONE",    "text": "What is the procedural standard for appointing a guardian ad litem in Harwick probate cases?"},
    {"id": "Q20", "label": "NONE",    "text": "What are the tax implications of the estate transfer to a non-family beneficiary under Harwick state law?"},
]


def upload() -> None:
    payload = json.dumps({
        "filename": FILENAME,
        "text": DOCUMENT_TEXT.strip(),
        "document_type": "text",
        "user_id": USER_ID,
        "conversation_id": CONVERSATION_ID,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/documents",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    print(f"  filename    : {FILENAME}")
    print(f"  document_id : {data['document_id']}")
    print(f"  chunks      : {data['chunks_created']}")
    print(f"  tokens      : {data['tokens_processed']}")


def ask(question: str) -> str:
    payload = json.dumps({
        "messages": [{"role": "user", "content": question}],
        "model": MODEL,
        "user_id": USER_ID,
        "conversation_id": CONVERSATION_ID,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    parts: list[str] = []
    with urllib.request.urlopen(req, timeout=90) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    parts.append(delta)
            except Exception:
                pass
    return "".join(parts).strip()


def main() -> None:
    print(f"\n{'='*60}")
    print("Rerank Threshold Analysis")
    print(f"Model       : {MODEL}")
    print(f"Conv ID     : {CONVERSATION_ID}")
    print(f"Delay       : {DELAY_SECONDS}s between questions")
    print(f"Total time  : ~{len(QUESTIONS) * DELAY_SECONDS // 60} min")
    print(f"{'='*60}\n")

    if not SKIP_UPLOAD:
        print("Uploading document...")
        try:
            upload()
        except Exception as e:
            print(f"UPLOAD FAILED: {e}")
            sys.exit(1)
        print("\nWaiting 3s before first question...\n")
        time.sleep(3)
    else:
        print(f"Using existing document: {FILENAME}")
        print(f"Conv ID: {CONVERSATION_ID}\n")

    for i, q in enumerate(QUESTIONS):
        print(f"{'─'*60}")
        print(f"{q['id']} | {q['label']}")
        print(f"Q: {q['text']}")
        try:
            answer = ask(q["text"])
            print(f"A: {answer}")
        except Exception as e:
            print(f"ERROR: {e}")

        if i < len(QUESTIONS) - 1:
            print(f"\n[{DELAY_SECONDS}s bekleniyor...]\n")
            time.sleep(DELAY_SECONDS)

    print(f"\n{'='*60}")
    print("Bitti. LangSmith'te her sorunun tool.rag_search")
    print("span'ini ac, Rerank-score degerlerini not al.")
    print(f"Conv ID: {CONVERSATION_ID}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
