"""Nepali entity pools and template strings for the synthetic reasoning set.

Mirrors ``hindi/reasoning_data/lexicon.py`` in structure exactly — same
domains, same task types, same four-surface-form-per-task-type design (L9
p.34), same gender-agreement handling on comparative adjectives (जेठो/जेठी,
कान्छो/कान्छी, अग्लो/अग्ली, होचो/होची, महँगो/महँगी, सस्तो/सस्ती) — but every
string is independently written natural Nepali, not a transliteration of
the Hindi, and entity pools are entirely separate names (Phase 1's
independence rule extends to this synthetic data too: no shared artifacts
between the two languages).
"""
from __future__ import annotations

from shared.reasoning.entities import EntityPool
from shared.reasoning.templates import Template, TemplateSet

ANSWER_PREFIX = "उत्तर: "

# ---------------------------------------------------------------- entities

_PERSON_GENDERS = {
    # 80 names (40 m / 40 f) — enlarged from an original 32-name pool for
    # the same reason as hindi/reasoning_data/lexicon.py: a small pool let
    # the model memorize identity->answer mappings instead of the relation
    # (course's own cautionary case, L9 p.22).
    "राम": "m", "श्याम": "m", "हरि": "m", "विष्णु": "m", "कृष्ण": "m",
    "विजय": "m", "अजय": "m", "बिनोद": "m", "सन्तोष": "m", "अनिल": "m",
    "सुरेश": "m", "महेश": "m", "दिनेश": "m", "रमेश": "m", "अमित": "m",
    "सुमित": "m", "विकास": "m", "आकाश": "m", "रोहित": "m", "मोहन": "m",
    "संजय": "m", "प्रकाश": "m", "राकेश": "m", "गणेश": "m", "करण": "m",
    "निखिल": "m", "पवन": "m", "नरेश": "m", "जगदीश": "m", "हरीश": "m",
    "यश": "m", "अरुण": "m", "वरुण": "m", "तरुण": "m", "गोपाल": "m",
    "मुकेश": "m", "राजेश": "m", "संदीप": "m", "दीपक": "m", "किशोर": "m",
    "सीता": "f", "गीता": "f", "राधा": "f", "कमला": "f", "सुनीता": "f",
    "मीना": "f", "प्रिया": "f", "पूजा": "f", "नेहा": "f", "स्नेहा": "f",
    "ललिता": "f", "ममता": "f", "अनीता": "f", "सरिता": "f", "रीता": "f",
    "सुमन": "f", "शीला": "f", "उर्मिला": "f", "विमला": "f", "आरती": "f",
    "ज्योति": "f", "किरण": "f", "मंजू": "f", "रेखा": "f", "सरोज": "f",
    "सुधा": "f", "इंदु": "f", "कुसुम": "f", "लता": "f", "मधु": "f",
    "निर्मला": "f", "शान्ति": "f", "गायत्री": "f", "पार्वती": "f", "दीपा": "f",
    "अंजली": "f", "प्रीति": "f", "स्वाती": "f", "वन्दना": "f", "सविता": "f",
}
PERSON_POOL = EntityPool(
    category="person", names=tuple(_PERSON_GENDERS), genders=_PERSON_GENDERS,
)

_OBJECT_GENDERS = {
    # 80 common nouns, same enlargement rationale.
    "स्याउ": "m", "केरा": "m", "अंगूर": "m", "सुन्तला": "m", "आँप": "m",
    "अनार": "m", "मेवा": "m", "तरबुजा": "m", "अम्बा": "m", "नरिवल": "m",
    "नासपाती": "f", "काफल": "m", "लिची": "f", "आरुबखडा": "m", "खरबुजा": "m",
    "भुइँकटहर": "m", "बयर": "m", "जामुन": "m", "स्याउल": "m", "रुखकटहर": "m",
    "गहुँ": "m", "चामल": "m", "दाल": "f", "आलु": "m", "प्याज": "m",
    "गोलभेडा": "m", "गाजर": "f", "मटर": "f", "पालुङ्गो": "m", "भिन्डी": "f",
    "भण्टा": "m", "काक्रो": "m", "मुला": "f", "लौका": "f", "करेला": "m",
    "मकै": "m", "जौ": "m", "कोदो": "m", "चना": "m", "सिमी": "f",
    "किताब": "f", "कलम": "f", "झोला": "m", "साइकल": "f", "घडी": "f",
    "मोबाइल": "m", "जुत्ता": "m", "छाता": "m", "चश्मा": "m", "पर्स": "m",
    "टोपी": "f", "ज्याकेट": "f", "स्कार्फ": "m", "बेल्ट": "m", "कुर्सी": "f",
    "टेबल": "f", "दराज": "m", "ऐना": "m", "सिरानी": "m", "कम्बल": "m",
    "पंखा": "m", "लालटिन": "f", "थैलो": "m", "बट्टा": "m", "भाँडो": "m",
    "चक्कु": "m", "चम्चा": "m", "गिलास": "m", "थाल": "m", "कचौरा": "m",
    "डोको": "m", "सुटकेस": "m", "कंघी": "f", "साबुन": "m", "तौलिया": "m",
    "सिसाकलम": "f", "रबर": "m", "स्केल": "m", "फाइल": "f", "डायरी": "f",
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


# --------------------------------------------------------------- pairwise
# want in {"smaller", "larger"}
# fields: name_a, name_b, value_a, value_b, gender_a, gender_b

def _pairwise_templates(more_m: str, more_f: str, less_m: str, less_f: str,
                        smaller_q: str, larger_q: str) -> TemplateSet:
    def _rel(f):
        is_more = f["value_a"] > f["value_b"]
        if f["gender_a"] == "m":
            return more_m if is_more else less_m
        return more_f if is_more else less_f

    def t0(**f):
        rel = _rel(f)
        q = smaller_q if f["want"] == "smaller" else larger_q
        return (f"{f['name_a']} {f['name_b']}भन्दा {rel} छ। {q}", _ans())

    def t1(**f):
        rel = _rel(f)
        q = smaller_q if f["want"] == "smaller" else larger_q
        return (f"यदि {f['name_a']} {f['name_b']}भन्दा {rel} छ भने, {q}", _ans())

    def t2(**f):
        rel = _rel(f)
        q = smaller_q if f["want"] == "smaller" else larger_q
        return (f"दिइएको छ: {f['name_a']} {f['name_b']}भन्दा {rel} छ। "
                f"भन्नुहोस्, {q}", _ans())

    def t3(**f):
        # held-out surface form
        rel = _rel(f)
        q = smaller_q if f["want"] == "smaller" else larger_q
        return (f"तुलना: {f['name_a']} र {f['name_b']}। {f['name_a']} "
                f"{f['name_b']}भन्दा {rel} छ। प्रश्न — {q}", _ans())

    return TemplateSet(
        task_type="pairwise",
        templates=(Template(0, t0), Template(1, t1), Template(2, t2), Template(3, t3)),
    )


# Question words use "को हो"/"कुन हो" — gender-invariant Nepali
# interrogatives — so only the stated relation adjective needs agreement
# (handled in _rel above); no separate masculine/feminine question form
# is needed.
PAIRWISE_TEMPLATES = {
    "age": _pairwise_templates(
        more_m="जेठो", more_f="जेठी", less_m="कान्छो", less_f="कान्छी",
        smaller_q="यीमध्ये कान्छो को हो?", larger_q="यीमध्ये जेठो को हो?"),
    "height": _pairwise_templates(
        more_m="अग्लो", more_f="अग्ली", less_m="होचो", less_f="होची",
        smaller_q="यीमध्ये होचो को हो?", larger_q="यीमध्ये अग्लो को हो?"),
    "price": _pairwise_templates(
        more_m="महँगो", more_f="महँगी", less_m="सस्तो", less_f="सस्ती",
        smaller_q="यीमध्ये सस्तो कुन हो?", larger_q="यीमध्ये महँगो कुन हो?"),
    "quantity": _pairwise_templates(
        more_m="धेरै", more_f="धेरै", less_m="थोरै", less_f="थोरै",
        smaller_q="यीमध्ये कसको मात्रा थोरै छ?", larger_q="यीमध्ये कसको मात्रा धेरै छ?"),
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
        return (f"{f['name_a']} {f['name_b']}भन्दा {rel_ab} छ। "
                f"{f['name_b']} {f['name_c']}भन्दा {rel_bc} छ। {q}", _ans())

    def t1(**f):
        rel_ab, rel_bc = _facts(f)
        q = _question(f["want"])
        return (f"पहिलो कुरा: {f['name_a']} {f['name_b']}भन्दा {rel_ab} छ। "
                f"दोस्रो कुरा: {f['name_b']} {f['name_c']}भन्दा {rel_bc} छ। {q}", _ans())

    def t2(**f):
        rel_ab, rel_bc = _facts(f)
        q = _question(f["want"])
        return (f"तीन नाम छन् — {f['name_a']}, {f['name_b']} र {f['name_c']}। "
                f"{f['name_a']} {f['name_b']}भन्दा {rel_ab} छ, र "
                f"{f['name_b']} {f['name_c']}भन्दा {rel_bc} छ। {q}", _ans())

    def t3(**f):
        # held-out surface form
        rel_ab, rel_bc = _facts(f)
        q = _question(f["want"])
        return (f"जानकारी: {f['name_a']} विरुद्ध {f['name_b']} — {rel_ab}; "
                f"{f['name_b']} विरुद्ध {f['name_c']} — {rel_bc}। "
                f"यसको आधारमा भन्नुहोस्: {q}", _ans())

    return TemplateSet(
        task_type="transitive3",
        templates=(Template(0, t0), Template(1, t1), Template(2, t2), Template(3, t3)),
    )


TRANSITIVE_TEMPLATES = {
    "age": _transitive_templates(
        more_m="जेठो", more_f="जेठी", less_m="कान्छो", less_f="कान्छी",
        smallest_q="यीमध्ये सबैभन्दा कान्छो को हो?",
        largest_q="यीमध्ये सबैभन्दा जेठो को हो?",
        middle_q="यीमध्ये बीचको को हो?"),
    "height": _transitive_templates(
        more_m="अग्लो", more_f="अग्ली", less_m="होचो", less_f="होची",
        smallest_q="यीमध्ये सबैभन्दा होचो को हो?",
        largest_q="यीमध्ये सबैभन्दा अग्लो को हो?",
        middle_q="यीमध्ये बीचको को हो?"),
    "price": _transitive_templates(
        more_m="महँगो", more_f="महँगी", less_m="सस्तो", less_f="सस्ती",
        smallest_q="यीमध्ये सबैभन्दा सस्तो कुन हो?",
        largest_q="यीमध्ये सबैभन्दा महँगो कुन हो?",
        middle_q="यीमध्ये बीचको मूल्य कुनको हो?"),
    "quantity": _transitive_templates(
        more_m="धेरै", more_f="धेरै", less_m="थोरै", less_f="थोरै",
        smallest_q="यीमध्ये सबैभन्दा थोरै मात्रा कसको हो?",
        largest_q="यीमध्ये सबैभन्दा धेरै मात्रा कसको हो?",
        middle_q="यीमध्ये बीचको मात्रा कसको हो?"),
}
