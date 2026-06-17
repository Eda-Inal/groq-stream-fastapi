# Sherlock Holmes — Agentic RAG Test Questions

Yüklenen: `Sherlock Holmes.txt` ilk %15 — 104 chunk, ~27,600 token  
user_id: `sherlock-rag-test`  
document_id: 103

---

## 1. Single-Retrieval Questions

> Tek bir chunk'tan yanıtlanabilir sorular.  
> Amaç: RAG pipeline'ının doğru chunk'ı bulup bulmadığını doğrulamak.  
> Zorluk: Orta — çok bariz anahtar kelime eşleşmesi olmasın ama cevap kesin olsun.

**Q1.** What specific physical clues does Holmes use to deduce that Watson has recently returned to medical practice?
- *Source section:* Early — Baker Street reunion scene
- *Expected answer:* Iodoform smell, nitrate of silver stain on right forefinger, stethoscope bulge in top hat

**Q2.** How does Holmes identify the anonymous note's author as German before the visitor even arrives?
- *Source section:* Early — Holmes and Watson examine the note
- *Expected answer:* Unusual sentence construction ("This account of you we have from all quarters received") — a German grammatical pattern; paper traced to Egria, Bohemia

**Q3.** What is the exact role Watson is asked to play in Holmes's plan at Briony Lodge?
- *Source section:* Middle — Holmes briefs Watson over cold beef and beer
- *Expected answer:* Station near the open sitting-room window, throw in a plumber's smoke-rocket on Holmes's hand signal, then shout "Fire"

**Q4.** In what disguise does Holmes arrive at Briony Lodge, and what does Watson say would have been lost to the stage?
- *Source section:* Middle — Holmes changes costume before leaving Baker Street
- *Expected answer:* An amiable Nonconformist clergyman; Watson says "The stage lost a fine actor"

**Q5.** Who founded the League of the Red-Headed Men, what was his nationality, and what condition did he set for eligibility?
- *Source section:* Late — Mr. Wilson recounts the advertisement
- *Expected answer:* Ezekiah Hopkins, American millionaire; restricted to London-resident grown men with real bright, blazing, fiery red hair

---

## 2. Multi-Hop Questions

> Cevap en az 2 farklı bölümden bilgi birleştirmeyi gerektirir.  
> Parantez içindeki bölümler kaynak section'ları gösterir.

**Q6.** Holmes deduces Watson's profession from objects on his person, and later does the same for Jabez Wilson. What specific object on Watson reveals he is a doctor, and what object on Wilson's body — combined with a second object on his watch chain — reveals he has been to China?
- *Sections:* Baker Street reunion + Wilson's visit (far apart in text)
- *Expected answer:* Watson: stethoscope bulge in top hat; Wilson: fish tattoo above right wrist + Chinese coin on watch chain

**Q7.** The King of Bohemia arrives wearing a mask, claiming to be Count Von Kramm. Holmes disguises himself as a Nonconformist clergyman for the Briony Lodge operation. What is the stated purpose of each person's disguise, and in which case does the disguise actually succeed?
- *Sections:* King's visit + Briony Lodge preparation and execution
- *Expected answer:* King: hide his identity as the agent of a royal; Holmes: get carried inside Briony Lodge as an "injured" man. Holmes's disguise succeeds; the King's fails because Holmes already knows who he is.

**Q8.** Watson describes Holmes between cases as alternating between two contrasting states. When Jabez Wilson brings the Red-Headed League problem, Holmes reacts very differently to having a puzzle in front of him. What are the two states Watson describes Holmes cycling through in the Scandal in Bohemia opening, and what specific phrase does Holmes use in the Red-Headed League case to signal he is now fully engaged in thinking?
- *Sections:* A Scandal in Bohemia opening (~chars 3,000) + Red-Headed League closing (~chars 83,000)
- *Expected answer:* "alternating from week to week between cocaine and ambition, the drowsiness of the drug, and the fierce energy of his own keen nature." Engaged state: "It is quite a three pipe problem, and I beg that you won't speak to me for fifty minutes."

**Q9.** Holmes lectures Watson that "You see, but you do not observe," using the Baker Street stairs as his example. How does this same principle apply when Holmes questions Wilson about Vincent Spaulding — what physical detail does Wilson openly admit he noticed, and what is Holmes's reaction that signals its true importance?
- *Sections:* Observation lecture with Watson + Spaulding interrogation at end of Wilson's visit
- *Expected answer:* Watson saw the stairs hundreds of times but never counted them (17 steps). Wilson confirms Spaulding's ears are pierced for earrings but assigns it no significance; Holmes sits up "in considerable excitement" saying "I thought as much," signalling Spaulding's real identity.

**Q10.** Holmes states a methodological principle when examining the anonymous note before the King of Bohemia arrives, and restates a related but different principle after Jabez Wilson leaves. What does Holmes say about theorizing in the first instance, and what does he say about bizarre cases in the second?
- *Sections:* A Scandal in Bohemia early (~chars 9,000) + Red-Headed League closing (~chars 82,000)
- *Expected answer:* First: "It is a capital mistake to theorise before one has data. Insensibly one begins to twist facts to suit theories, instead of theories to suit facts." Second: "As a rule, the more bizarre a thing is the less mysterious it proves to be. It is your commonplace, featureless crimes which are really puzzling."

**Q11.** Two different characters in this text introduce themselves or are introduced under false names. What false name and title does the King of Bohemia give when he first enters Holmes's rooms, and what false name does the man who managed the Red-Headed League office give to the building's landlord when later confronted by Wilson?
- *Sections:* A Scandal in Bohemia (~chars 12,000) + Red-Headed League (~chars 79,000)
- *Expected answer:* The King introduces himself as "Count Von Kramm, a Bohemian nobleman." The League office manager told the landlord his name was William Morris, described as a solicitor using the room as a temporary convenience.

**Q12.** Holmes says after the Briony Lodge operation "I know where it is" about the photograph, but does not yet have it. When Wilson arrives with a completely different case, Holmes tells him "graver issues hang from it than might at first sight appear." What does Wilson consider to be the gravest consequence of the League affair, and how does Holmes's framing of it differ?
- *Sections:* Post-Briony Lodge debrief + Holmes's response to Wilson's story
- *Expected answer:* Wilson's gravest concern is losing four pounds a week. Holmes frames the case as having implications beyond Wilson's personal loss, hinting at a larger criminal scheme — which contrasts sharply with Wilson's narrow financial grievance.

**Q13.** Holmes makes two deductions that prove wrong or incomplete within this text. He deduces Watson has put on "seven and a half pounds" but Watson corrects him to seven. He also tells the King he already knew who he was. What does the first moment reveal about how Holmes handles being corrected, and what does the second reveal about the value he places on surprise?
- *Sections:* Watson reunion (weight deduction) + King's unmasking scene
- *Expected answer:* Holmes accepts Watson's correction without embarrassment ("I should have thought a little more"). With the King, Holmes demonstrates foreknowledge calmly and dryly, valuing the effect of appearing to already know everything.

**Q14.** Watson attempts to read Jabez Wilson using Holmes's deductive method before Holmes speaks, and had done the same earlier with the anonymous note sent by the King of Bohemia's agent. What conclusion does Watson reach in each attempt, and how does the text signal whether he succeeds or fails each time?
- *Sections:* A Scandal in Bohemia early (~chars 8,000) + Red-Headed League opening (~chars 51,000)
- *Expected answer:* With the note: Watson deduces the writer is "presumably well to do" from the expensive paper — Holmes confirms this and builds further on it, so Watson partially succeeds. With Wilson: Watson says "I did not gain very much, however, by my inspection" and describes only obvious surface details (red hair, chagrin), while Holmes immediately rattles off five facts Watson missed entirely.

**Q15.** Holmes demonstrates his deductive method twice in quick succession when Wilson visits: first without explanation, then step by step when asked. What does Holmes say immediately after explaining everything that reveals his ambivalence about transparency, and how does Watson's earlier reaction to Holmes's explanations foreshadow this?
- *Sections:* Watson reunion ("ridiculously simple") + Wilson's visit ("Omne ignotum pro magnifico")
- *Expected answer:* After explaining to Wilson, Holmes says "I begin to think, Watson, that I make a mistake in explaining — Omne ignotum pro magnifico — my poor little reputation will suffer shipwreck if I am so candid." Watson had earlier said Holmes's explanations always make the thing seem ridiculously simple, which is exactly the reputation damage Holmes now fears.

---

## 3. Hallucination Traps

> Bu sorularda doğru cevap ya yüklenen %15'in dışında kalır ya da metin onu hiç vermez.  
> Beklenen davranış: model "This information was not found in your documents." demeli.  
> Eğer özgün bir cevap üretirse → hallucination.

**H1.** How does Holmes ultimately retrieve the photograph from Irene Adler at the end of the Scandal in Bohemia case?
- *Trap:* He never retrieves it. Adler outsmarts Holmes and leaves with the photograph. The uploaded text ends before any resolution is shown.
- *Expected behavior:* Model should state the outcome is not in the documents, or — if it reaches the resolution — correctly say Holmes does not get the photo. If it invents a retrieval scene, it is hallucinating.

**H2.** What is the physical description of Irene Adler's appearance — her hair color, height, and facial features — as given in the text?
- *Trap:* The uploaded text only describes her as having a "superb figure" and refers to her as a "beautiful creature." No hair color, height, or facial details are given.
- *Expected behavior:* Model should say the document does not provide a physical description beyond a general mention of her figure.

**H3.** In The Adventure of the Speckled Band, what does Holmes identify as the murder weapon and how is it delivered to the victim?
- *Trap:* This story (chapter VIII) is entirely outside the uploaded 15%. The model has strong training knowledge about it (a snake through a ventilator).
- *Expected behavior:* Model should say this content is not in the uploaded documents. If it answers from training knowledge, it is hallucinating from outside the knowledge base.
