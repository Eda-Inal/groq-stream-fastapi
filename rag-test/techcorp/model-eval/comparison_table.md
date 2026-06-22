# TechCorp — Model Generation Eval

Score = keyword match ratio (pass ≥ 0.8). For `absent_plausible`: not_found=yes means model correctly said 'not in document'.

## run_001 — 2026-06-22 14:08 | llama-3.3-70b-versatile

**Pass: 1/1 (100%) | RAG called: 0/1 | Avg keyword score: 1.0**


| Q | Diff | Category | Pass | Score | RAG Query | Model Answer | Expected Answer |
|---|---|---|---|---|---|---|---|
| Q01 | easy | exact_match | ✅ | 1.0 | — | All remote work must be conducted over TechCorp's VPN when accessing internal systems. Source: techc… | All remote work must be conducted over TechCorp's VPN when accessing internal sy… |

## run_002 — 2026-06-22 14:17 | meta-llama/llama-4-scout-17b-16e-instruct

**Pass: 1/1 (100%) | RAG called: 0/1 | Avg keyword score: 1.0**


| Q | Diff | Category | Pass | Score | RAG Query | Model Answer | Expected Answer |
|---|---|---|---|---|---|---|---|
| Q01 | easy | exact_match | ✅ | 1.0 | — | All remote work must be conducted over TechCorp's VPN when accessing internal systems.  Source: tech… | All remote work must be conducted over TechCorp's VPN when accessing internal sy… |

## run_003 — 2026-06-22 14:57 | meta-llama/llama-4-scout-17b-16e-instruct

**Pass: 1/1 (100%) | RAG called: 1/1 | Avg keyword score: 1.0**


| Q | Diff | Category | Pass | Score | RAG Query | Model Answer | Expected Answer |
|---|---|---|---|---|---|---|---|
| Q01 | easy | exact_match | ✅ | 1.0 | remote work internal systems access requirements | All remote work must be conducted over TechCorp's VPN when accessing internal systems.  Source: tech… | All remote work must be conducted over TechCorp's VPN when accessing internal sy… |

## run_005 — 2026-06-22 15:10 | meta-llama/llama-4-scout-17b-16e-instruct

**Pass: 5/10 (50%) | RAG called: 9/10 | Avg keyword score: 0.817**


| Q | Question | Diff | Category | Pass | Score | RAG Query | Model Answer | Expected Answer |
|---|---|---|---|---|---|---|---|---|
| Q01 | What must all remote work be conducted over when accessing internal sy… | easy | exact_match | ✅ | 1.0 | remote work internal systems access requirements | All remote work must be conducted over TechCorp's VPN when accessing internal systems.  Source: tech… | All remote work must be conducted over TechCorp's VPN when accessing internal sy… |
| Q02 | Within how many days must business expenses be submitted through the e… | easy | exact_match | ❌ | 0.33 | — | rag_search(query="expense submission deadline") | All business expenses must be submitted within 30 days of the expense being incu… |
| Q03 | How much notice must an employee provide in writing before resigning? | easy | exact_match | ✅ | 1.0 | notice period for resignation | The employee must provide at least two weeks' written notice to their manager before resigning.  Sou… | Employees must provide at least two weeks written notice to their manager before… |
| Q04 | What is the email address employees should contact if their device is … | easy | named_entity | ❌ | 0.75 | email address for lost or stolen device | The email address employees should contact if their device is lost or stolen is security@techcorp.co… | If a device is lost or stolen, employees must report it immediately to the IT Se… |
| Q05 | What is the home office stipend amount provided to new employees? | easy | named_entity | ✅ | 1.0 | home office stipend amount for new employees | The home office stipend amount provided to new employees is $1,500 USD. This stipend is to be used w… | TechCorp provides a home office stipend of $1,500 USD to all new employees, to b… |
| Q06 | How long is the vesting period for stock options at TechCorp? | easy | direct_fact | ❌ | 0.75 | TechCorp stock options vesting period | The vesting period for stock options at TechCorp is four years, with a one-year cliff. This means th… | Stock options vest over a four-year period, with a one-year cliff. This means 25… |
| Q07 | How many days of paid bereavement leave does TechCorp provide? | easy | direct_fact | ✅ | 1.0 | TechCorp paid bereavement leave days | TechCorp provides three days of paid bereavement leave following the death of an immediate family me… | TechCorp provides three days of paid bereavement leave following the death of an… |
| Q08 | What is the annual professional development budget per employee? | easy | direct_fact | ❌ | 0.67 | annual professional development budget per employee | The annual professional development budget per employee is $2,000 USD. This budget can be used for c… | TechCorp offers a professional development budget of $2,000 USD per employee per… |
| Q09 | When can an employee begin taking parental leave after a child is born… | easy | dense_primary | ✅ | 1.0 | When can an employee begin taking parental leave after … | Leave can be taken at any point within the first 12 months following the birth, adoption, or foster … | Parental leave can be taken at any point within the first 12 months following th… |
| Q10 | What happens to unexercised stock options after an employee leaves Tec… | easy | dense_primary | ❌ | 0.67 | unexercised stock options after leaving TechCorp | After an employee leaves TechCorp, unexercised stock options expire 90 days after their last day of … | After leaving TechCorp, employees have 90 days to exercise their vested options.… |
