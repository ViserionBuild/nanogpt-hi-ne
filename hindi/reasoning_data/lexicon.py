"""Hindi entity pools and template strings for the synthetic reasoning set.

All text is natural Hindi phrasing in Devanagari script (spec 3.1: "not
English prompts with foreign names only"), including grammatical-gender
agreement on comparative adjectives (बड़ा/बड़ी, छोटा/छोटी, लंबा/लंबी,
महंगा/महंगी, सस्ता/सस्ती) — every person name and object noun is tagged
with its grammatical gender in the pool, and templates pick the agreeing
adjective form per entity rather than defaulting to masculine throughout,
which would read as ungrammatical to a Hindi speaker.

Four domains, each with a pairwise and a 3-entity transitive task type,
each with four surface-form templates (course L9 p.34: write four, train
on three, hold the fourth out — here, template_id 3 in every TemplateSet).

Answer format: every answer is exactly ``"उत्तर: <name>"`` (Answer: <name>)
followed by the tokenizer's own ``<eos>``. The explicit "उत्तर:" marker is
the course's stop-token/delimiter point (L9 p.9): the model needs a clear
boundary between "the question has been fully stated" and "now produce the
answer", or it has no signal for when to stop reasoning and start
answering. ``<eos>`` (already token id 3 from Phase 1) is what stops
generation; the marker is what the loss mask keys off inside the answer
span.
"""
from __future__ import annotations

from shared.reasoning.entities import EntityPool
from shared.reasoning.templates import Template, TemplateSet

ANSWER_PREFIX = "उत्तर: "

# ---------------------------------------------------------------- entities
# gender: "m" (masculine) or "f" (feminine) — used to pick the agreeing
# comparative-adjective form (see _adj below).

_PERSON_GENDERS = {
    # 80 names (40 m / 40 f). Deliberately this large — a 24-name pool let
    # the model shortcut to memorized identity->answer mappings instead of
    # reading the stated relation (course's own cautionary case, L9 p.22):
    # at 2,400 training examples, 24 names means ~100 appearances each,
    # cheap to memorize; 80 names cuts that to ~30 and makes memorization
    # a worse strategy than actually reading the comparison.
    "राम": "m", "श्याम": "m", "मोहन": "m", "अर्जुन": "m", "भीम": "m",
    "विजय": "m", "अजय": "m", "राजू": "m", "सोनू": "m", "अनिल": "m",
    "सुरेश": "m", "महेश": "m", "दिनेश": "m", "रमेश": "m", "अमित": "m",
    "सुमित": "m", "विकास": "m", "आकाश": "m", "रोहित": "m", "मोहित": "m",
    "संजय": "m", "प्रकाश": "m", "राकेश": "m", "गणेश": "m", "करण": "m",
    "अभिषेक": "m", "निखिल": "m", "पवन": "m", "नरेश": "m", "जगदीश": "m",
    "हरीश": "m", "देवेंद्र": "m", "यश": "m", "अरुण": "m", "वरुण": "m",
    "तरुण": "m", "गोपाल": "m", "मुकेश": "m", "राजेश": "m", "संदीप": "m",
    "सीता": "f", "गीता": "f", "राधा": "f", "कविता": "f", "सुनीता": "f",
    "मीना": "f", "प्रिया": "f", "पूजा": "f", "नेहा": "f", "स्नेहा": "f",
    "ललिता": "f", "ममता": "f", "अनीता": "f", "सरिता": "f", "रीता": "f",
    "सुमन": "f", "कमला": "f", "शीला": "f", "उर्मिला": "f", "विमला": "f",
    "आरती": "f", "ज्योति": "f", "किरण": "f", "मंजू": "f", "रेखा": "f",
    "सरोज": "f", "सुधा": "f", "इंदु": "f", "कुसुम": "f", "लता": "f",
    "मधु": "f", "निर्मला": "f", "शांति": "f", "गायत्री": "f", "पार्वती": "f",
    "दीपा": "f", "अंजलि": "f", "प्रीति": "f", "स्वाति": "f", "वंदना": "f",
}
PERSON_POOL = EntityPool(
    category="person", names=tuple(_PERSON_GENDERS), genders=_PERSON_GENDERS,
)

_OBJECT_GENDERS = {
    # 80 common nouns (fruits/vegetables/grains/household items), same
    # enlargement rationale as _PERSON_GENDERS above.
    "सेब": "m", "केला": "m", "अंगूर": "m", "संतरा": "m", "आम": "m",
    "अनार": "m", "पपीता": "m", "तरबूज़": "m", "अमरूद": "m", "नारियल": "m",
    "नाशपाती": "f", "चीकू": "m", "लीची": "f", "आड़ू": "m", "खरबूजा": "m",
    "अनानास": "m", "बेर": "m", "जामुन": "m", "शरीफा": "m", "कटहल": "m",
    "गेहूं": "m", "चावल": "m", "दाल": "f", "आलू": "m", "प्याज़": "m",
    "टमाटर": "m", "गाजर": "f", "मटर": "f", "पालक": "m", "भिंडी": "f",
    "बैंगन": "m", "खीरा": "m", "मूली": "f", "लौकी": "f", "करेला": "m",
    "मक्का": "m", "जौ": "m", "बाजरा": "m", "चना": "m", "राजमा": "m",
    "किताब": "f", "कलम": "f", "बैग": "m", "साइकिल": "f", "घड़ी": "f",
    "मोबाइल": "m", "जूता": "m", "छाता": "m", "चश्मा": "m", "पर्स": "m",
    "टोपी": "f", "जैकेट": "f", "स्कार्फ़": "m", "बेल्ट": "m", "कुर्सी": "f",
    "मेज़": "f", "अलमारी": "f", "दर्पण": "m", "तकिया": "m", "कंबल": "m",
    "पंखा": "m", "लालटेन": "f", "थैला": "m", "डिब्बा": "m", "बर्तन": "m",
    "चाकू": "m", "चम्मच": "m", "गिलास": "m", "थाली": "f", "कटोरी": "f",
    "टोकरी": "f", "सूटकेस": "m", "कंघी": "f", "साबुन": "m", "तौलिया": "m",
    "पेंसिल": "f", "रबड़": "f", "स्केल": "m", "फ़ाइल": "f", "डायरी": "f",
}
OBJECT_POOL = EntityPool(
    category="object", names=tuple(_OBJECT_GENDERS), genders=_OBJECT_GENDERS,
)

# ------------------------------------------------------------------ domains

DOMAINS = {
    "age": {"pool": PERSON_POOL, "unit_low": 5, "unit_high": 80},
    "height": {"pool": PERSON_POOL, "unit_low": 90, "unit_high": 200},
    "price": {"pool": OBJECT_POOL, "unit_low": 10, "unit_high": 500},
    "quantity": {"pool": OBJECT_POOL, "unit_low": 1, "unit_high": 100},
}


def _ans(label_field: str = "answer") -> str:
    return ANSWER_PREFIX + "{" + label_field + "}"


def _adj(stem_m: str, stem_f: str, gender: str) -> str:
    """Pick the gender-agreeing form of an -आ/-ई adjective stem.

    ``stem_m`` is the masculine surface form (e.g. "बड़ा"), ``stem_f`` the
    feminine (e.g. "बड़ी"). Words like "ज़्यादा"/"कम" that don't inflect are
    passed identically for both and this is a no-op.
    """
    return stem_m if gender == "m" else stem_f


# --------------------------------------------------------------- pairwise
# want in {"smaller", "larger"}
# fields: name_a, name_b, value_a, value_b, gender_a, gender_b

def _pairwise_templates(more_m: str, more_f: str, less_m: str, less_f: str,
                        smaller_q_m: str, smaller_q_f: str,
                        larger_q_m: str, larger_q_f: str) -> TemplateSet:
    def _rel_and_q(f):
        """Relation word agrees with entity A's gender (A is the sentence
        subject); the question word agrees with the gender of whichever
        entity the label ends up being (resolved after sampling, so the
        question clause here uses a gender-neutral fallback and the
        per-example gender is threaded through want/answer at label time).
        Since the label's identity is only known after comparing values,
        the question clause uses BOTH genders' vocabulary only when they
        coincide; to keep grammar correct without a second templating pass,
        the question is phrased with the ambiguous-gender interrogative
        "कौन" (who / which one) — this is why smaller_q/larger_q are
        already word-final ("कौन है?") and gender-invariant by design; the
        gendered slot that actually needs agreement is the *relation*
        adjective, which is resolved here.
        """
        rel = (more_m if f["value_a"] > f["value_b"] else less_m) \
            if f["gender_a"] == "m" else \
            (more_f if f["value_a"] > f["value_b"] else less_f)
        return rel

    def t0(**f):
        rel = _rel_and_q(f)
        q = smaller_q_m if f["want"] == "smaller" else larger_q_m
        return (f"{f['name_a']} {f['name_b']} से {rel} है। {q}", _ans())

    def t1(**f):
        rel = _rel_and_q(f)
        q = smaller_q_m if f["want"] == "smaller" else larger_q_m
        return (f"अगर {f['name_a']} {f['name_b']} से {rel} है, तो {q}", _ans())

    def t2(**f):
        rel = _rel_and_q(f)
        q = smaller_q_m if f["want"] == "smaller" else larger_q_m
        return (f"दिया गया है: {f['name_a']} {f['name_b']} से {rel} है। "
                f"बताइए, {q}", _ans())

    def t3(**f):
        # held-out surface form
        rel = _rel_and_q(f)
        q = smaller_q_m if f["want"] == "smaller" else larger_q_m
        return (f"तुलना: {f['name_a']} और {f['name_b']}। {f['name_a']} "
                f"{f['name_b']} से {rel} है। प्रश्न — {q}", _ans())

    return TemplateSet(
        task_type="pairwise",
        templates=(Template(0, t0), Template(1, t1), Template(2, t2), Template(3, t3)),
    )


# Question words below use "कौन"/"कौन सा" which are gender-invariant
# interrogatives in Hindi ("who"/"which one"), so no gender variant is
# needed for the *question* clause — only the stated relation needs
# agreement, handled in _rel_and_q above.
PAIRWISE_TEMPLATES = {
    "age": _pairwise_templates(
        more_m="बड़ा", more_f="बड़ी", less_m="छोटा", less_f="छोटी",
        smaller_q_m="इनमें से छोटा कौन है?", smaller_q_f="इनमें से छोटा कौन है?",
        larger_q_m="इनमें से बड़ा कौन है?", larger_q_f="इनमें से बड़ा कौन है?"),
    "height": _pairwise_templates(
        more_m="लंबा", more_f="लंबी", less_m="छोटा", less_f="छोटी",
        smaller_q_m="इनमें से छोटा कौन है?", smaller_q_f="इनमें से छोटा कौन है?",
        larger_q_m="इनमें से लंबा कौन है?", larger_q_f="इनमें से लंबा कौन है?"),
    "price": _pairwise_templates(
        more_m="महंगा", more_f="महंगी", less_m="सस्ता", less_f="सस्ती",
        smaller_q_m="इनमें से सस्ता कौन सा है?", smaller_q_f="इनमें से सस्ता कौन सा है?",
        larger_q_m="इनमें से महंगा कौन सा है?", larger_q_f="इनमें से महंगा कौन सा है?"),
    "quantity": _pairwise_templates(
        more_m="ज़्यादा", more_f="ज़्यादा", less_m="कम", less_f="कम",
        smaller_q_m="इनमें से किसकी मात्रा कम है?", smaller_q_f="इनमें से किसकी मात्रा कम है?",
        larger_q_m="इनमें से किसकी मात्रा ज़्यादा है?", larger_q_f="इनमें से किसकी मात्रा ज़्यादा है?"),
}


# ------------------------------------------------------------ transitive3
# want in {"smallest", "largest", "middle"}
# fields: name_a, name_b, name_c, value_a, value_b, value_c,
#         gender_a, gender_b, gender_c

def _transitive_templates(more_m: str, more_f: str, less_m: str, less_f: str,
                          smallest_q: str, largest_q: str, middle_q: str) -> TemplateSet:
    def _rel(gender: str, is_more: bool) -> str:
        if gender == "m":
            return more_m if is_more else less_m
        return more_f if is_more else less_f

    def _facts(f):
        rel_ab = _rel(f["gender_a"], f["value_a"] > f["value_b"])
        rel_bc = _rel(f["gender_b"], f["value_b"] > f["value_c"])
        return rel_ab, rel_bc

    def _question(want):
        return {"smallest": smallest_q, "largest": largest_q,
                "middle": middle_q}[want]

    def t0(**f):
        rel_ab, rel_bc = _facts(f)
        q = _question(f["want"])
        return (f"{f['name_a']} {f['name_b']} से {rel_ab} है। "
                f"{f['name_b']} {f['name_c']} से {rel_bc} है। {q}", _ans())

    def t1(**f):
        rel_ab, rel_bc = _facts(f)
        q = _question(f["want"])
        return (f"पहली बात: {f['name_a']} {f['name_b']} से {rel_ab} है। "
                f"दूसरी बात: {f['name_b']} {f['name_c']} से {rel_bc} है। {q}", _ans())

    def t2(**f):
        rel_ab, rel_bc = _facts(f)
        q = _question(f["want"])
        return (f"तीन नाम हैं — {f['name_a']}, {f['name_b']} और {f['name_c']}। "
                f"{f['name_a']} {f['name_b']} से {rel_ab} है, और "
                f"{f['name_b']} {f['name_c']} से {rel_bc} है। {q}", _ans())

    def t3(**f):
        # held-out surface form
        rel_ab, rel_bc = _facts(f)
        q = _question(f["want"])
        return (f"जानकारी: {f['name_a']} बनाम {f['name_b']} — {rel_ab}; "
                f"{f['name_b']} बनाम {f['name_c']} — {rel_bc}। "
                f"इस आधार पर बताइए: {q}", _ans())

    return TemplateSet(
        task_type="transitive3",
        templates=(Template(0, t0), Template(1, t1), Template(2, t2), Template(3, t3)),
    )


# "सबसे छोटा/बड़ा कौन है?" — कौन is gender-invariant, so a single question
# form covers both genders (unlike the relation adjectives, resolved above).
TRANSITIVE_TEMPLATES = {
    "age": _transitive_templates(
        more_m="बड़ा", more_f="बड़ी", less_m="छोटा", less_f="छोटी",
        smallest_q="इनमें से सबसे छोटा कौन है?",
        largest_q="इनमें से सबसे बड़ा कौन है?",
        middle_q="इनमें से बीच वाला कौन है?"),
    "height": _transitive_templates(
        more_m="लंबा", more_f="लंबी", less_m="छोटा", less_f="छोटी",
        smallest_q="इनमें से सबसे छोटा कौन है?",
        largest_q="इनमें से सबसे लंबा कौन है?",
        middle_q="इनमें से बीच वाला कौन है?"),
    "price": _transitive_templates(
        more_m="महंगा", more_f="महंगी", less_m="सस्ता", less_f="सस्ती",
        smallest_q="इनमें से सबसे सस्ता कौन सा है?",
        largest_q="इनमें से सबसे महंगा कौन सा है?",
        middle_q="इनमें से बीच वाला दाम किसका है?"),
    "quantity": _transitive_templates(
        more_m="ज़्यादा", more_f="ज़्यादा", less_m="कम", less_f="कम",
        smallest_q="इनमें से सबसे कम मात्रा किसकी है?",
        largest_q="इनमें से सबसे ज़्यादा मात्रा किसकी है?",
        middle_q="इनमें से बीच वाली मात्रा किसकी है?"),
}
