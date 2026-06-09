# Visual chunking test script.
# Usage: .venv/Scripts/python scripts/chunk_inspect.py

import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.chunking import chunk_document, count_tokens

# ─── PASTE YOUR TEXTS HERE ────────────────────────────────────────────────────

TEXT_1 = """
Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight into chemical energy. Using light absorbed by chlorophyll — the green pigment found in chloroplasts — these organisms take in carbon dioxide from the air and water from the soil to produce glucose and oxygen. The overall reaction can be summarized as: six molecules of carbon dioxide plus six molecules of water, energized by light, yield one molecule of glucose and six molecules of oxygen. This process is the foundation of almost all life on Earth, as it provides the primary source of organic matter and the oxygen that most living organisms need to breathe..
"""

TEXT_2 = """
The Roman Republic, established around 509 BCE after the overthrow of the last Roman king, Tarquinius Superbus, represented one of the ancient world's most sophisticated systems of governance. Power was deliberately divided to prevent any single individual from gaining absolute control. At the top sat two consuls, elected annually, who shared executive authority and could veto each other's decisions — a system designed to mirror the checks and balances that modern democracies would later adopt centuries later. Below the consuls, the Senate served as the principal advisory and legislative body. Composed initially of patrician aristocrats, it gradually opened to plebeians through prolonged political struggle known as the Conflict of the Orders. This centuries-long tension between the privileged patrician class and the common plebeian population fundamentally shaped Roman law, producing landmark legislation such as the Twelve Tables around 450 BCE — the earliest attempt to codify Roman law in written form and make it accessible to all citizens. The Republic's military success was equally remarkable. Roman legions, disciplined and adaptable, expanded Roman territory across the Italian peninsula by the third century BCE. However, this very expansion carried the seeds of the Republic's eventual decline. The enormous influx of slaves from conquered territories undercut free Roman labor, widened wealth inequality, and created social tensions that reformers like Tiberius and Gaius Gracchus struggled to address — ultimately at the cost of their lives. By the first century BCE, civil wars between powerful generals like Marius, Sulla, Caesar, and later Octavian had fatally eroded the Republic's institutions, paving the way for the transition to the Roman Empire.
"""

TEXT_3 = """
The history of the internet is a story of scientific ambition, military necessity, and ultimately a radical transformation in how human beings communicate, work, and understand the world. Its origins lie not in Silicon Valley boardrooms but in Cold War anxiety. In 1957, the Soviet Union's launch of Sputnik shocked the United States government into action, leading to the creation of the Advanced Research Projects Agency, known as ARPA, in 1958. This agency would become the incubator for the technologies that eventually gave birth to the modern internet. The first major milestone was ARPANET, launched in 1969 under contract with the U.S. Department of Defense. The network's initial purpose was practical and strategic: create a communication system resilient enough to survive a nuclear attack by routing information through multiple paths rather than relying on a single point of failure. On October 29, 1969, the first message was sent from UCLA to the Stanford Research Institute. The intended message was "login," but the system crashed after just the first two letters — making "lo" the unintentionally poetic first words of the digital age. Throughout the 1970s, ARPANET expanded to connect universities and research institutions across the United States. A critical development came in 1974 when Vint Cerf and Bob Kahn published their paper describing the Transmission Control Protocol, or TCP — later combined with the Internet Protocol to form TCP/IP. This suite of communication rules became the universal language that allowed different computer networks to speak to one another regardless of their underlying hardware or software. The adoption of TCP/IP as the standard protocol on January 1, 1983 — a date some call the "birthday of the internet" — transformed a collection of disconnected networks into a single, interconnected whole. The 1980s saw the internet grow beyond military and academic circles, though it remained largely inaccessible to the general public. The introduction of the Domain Name System in 1984 replaced numerical IP addresses with human-readable names like mit.edu, making navigation far more intuitive. Meanwhile, email had already become a surprisingly dominant use case, accounting for the majority of ARPANET traffic even in its early years, hinting at the internet's social potential long before social media was conceived. The true democratization of the internet arrived with the World Wide Web, invented by British computer scientist Tim Berners-Lee at CERN in 1989 and made publicly available in 1991. Berners-Lee's innovation was deceptively simple: link documents together using hypertext so that any page could reference and connect to any other page. Combined with HTTP (the transfer protocol), HTML (the markup language), and URLs (the addressing system), the Web gave ordinary users an intuitive visual interface to navigate information. The first widely adopted browser, Mosaic, released in 1993, made the experience graphical and accessible. Within two years, internet usage was doubling every year. The late 1990s brought the dot-com boom, a period of frenzied investment and reckless optimism in which hundreds of internet startups attracted enormous capital before many collapsed in the dot-com bust of 2000–2001. Despite the economic carnage, the infrastructure built during this period — fiber optic cables laid across oceans, data centers constructed across continents — outlasted the companies that funded it and formed the backbone of the next phase of internet growth. The 2000s ushered in what is now called Web 2.0, characterized by user-generated content and social interaction. Wikipedia launched in 2001, demonstrating that collaborative knowledge creation could outpace traditional encyclopedias. Friendster, MySpace, and eventually Facebook transformed the internet into a social space. YouTube, founded in 2005 and acquired by Google in 2006, made video publishing accessible to anyone with a camera. Twitter, launched in 2006, compressed public discourse into 140 characters. The smartphone revolution, accelerated by the iPhone's debut in 2007, untethered the internet from the desktop entirely, making it a constant, ambient presence in daily life. Today, the internet connects an estimated five billion people across the globe and undergirds virtually every sector of the modern economy. Cloud computing has moved software and data off personal devices and into remote server farms. Machine learning models train on datasets scraped from internet content. Generative AI systems now assist in writing, coding, and creative work. Yet the internet's rapid growth has also intensified debates about privacy, misinformation, digital surveillance, and the concentration of power in a handful of technology companies. The network that began as a military research project is now, in many respects, the central nervous system of human civilization — and its governance, security, and future direction remain among the most contested questions of the twenty-first century.
"""

# ─── SETTINGS (optional, leave None to use .env defaults) ────────────────────

CHUNK_SIZE   = 200    # in tokens
OVERLAP      = 50     # in tokens
SHORT_DOC_MAX = 0     # 0 = always split (for observation)

# ─────────────────────────────────────────────────────────────────────────────

SEPARATOR = "-" * 60

def inspect(label: str, text: str) -> None:
    total_tokens = count_tokens(text.strip())
    print(f"\n{SEPARATOR}")
    print(f"  {label}  |  Total: {total_tokens} tokens  |  {len(text.strip())} chars")
    print(SEPARATOR)

    kwargs = {"short_doc_max_tokens": SHORT_DOC_MAX}
    if CHUNK_SIZE is not None:
        kwargs["chunk_size_tokens"] = CHUNK_SIZE
    if OVERLAP is not None:
        kwargs["overlap_tokens"] = OVERLAP

    try:
        chunks = chunk_document(text, **kwargs)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print(f"  -> {len(chunks)} chunks created\n")
    for i, c in enumerate(chunks):
        bar = "#" * min(c.token_count // 5, 60)
        print(f"  Chunk {i+1:02d}  [{c.token_count:>4} tokens]  {bar}")
        print(f"  {c.text[:200].replace(chr(10), ' | ')}")
        if len(c.text) > 200:
            print(f"  ... (+{len(c.text)-200} more chars)")
        print()

    if len(chunks) < 2:
        print("  [Need at least 2 chunks for overlap test]\n")
        return

    # --- SENTENCE BOUNDARY CHECK ---
    print("  --- SENTENCE BOUNDARY CHECK ---")
    sentence_end = {".", "!", "?", '."', '!"', '?"', "...", ":"}
    cut_count = 0
    for i, c in enumerate(chunks):
        text = c.text.strip()
        last_char = text[-1] if text else ""
        first_char = text[0] if text else ""
        ending_ok = any(text.endswith(e) for e in sentence_end) or text.endswith(("```", '"""', "'''"))
        starting_ok = first_char.isupper() or text.startswith(("-", "*", "#", "`", '"', "'"))
        status_end = "OK " if ending_ok else "CUT"
        status_start = "OK " if (i == 0 or starting_ok) else "CUT"
        if status_end == "CUT" or status_start == "CUT":
            cut_count += 1
        print(f"\n  Chunk {i+1:02d}  [end:{status_end}] [start:{status_start}]")
        print(f"    Last 60 chars : ...{text[-60:].replace(chr(10), ' ')}")
        print(f"    First 60 chars: {text[:60].replace(chr(10), ' ')}...")

    print(f"\n  Result: {cut_count} chunk(s) with possible sentence cuts / {len(chunks)} total")

    # --- OVERLAP CHECK ---
    print("\n  --- OVERLAP CHECK ---")
    from app.services.chunking import count_tokens as ct
    for i in range(len(chunks) - 1):
        prev_tail = " ".join(chunks[i].text.split()[-15:])
        next_head = " ".join(chunks[i + 1].text.split()[:15])
        shared = set(chunks[i].text.split()) & set(chunks[i + 1].text.split()[:60])
        overlap_token_est = ct(" ".join(shared))
        print(f"\n  Chunk {i+1} end  : ...{prev_tail}")
        print(f"  Chunk {i+2} start: {next_head}...")
        print(f"  Shared words: {len(shared)}  (~{overlap_token_est} tokens)")


if __name__ == "__main__":
    inspect("TEXT 1", TEXT_1)
    inspect("TEXT 2", TEXT_2)
    inspect("TEXT 3", TEXT_3)
    print(SEPARATOR)
    print("  Done.")
    print(SEPARATOR)
