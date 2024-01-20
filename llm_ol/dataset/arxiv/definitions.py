ALIASES = [
    ["math.MP", "math-ph"],
    ["stat.TH", "math.ST"],
    ["math.IT", "cs.IT"],
    ["econ.GN", "q-fin.EC"],
    ["cs.SY", "eess.SY"],
    ["cs.NA", "math.NA"],
    ["grp_physics", "physics"],
    ["grp_econ", "econ"],
    ["grp_math", "math"],
    ["grp_q-bio", "q-bio"],
    ["grp_q-fin", "q-fin"],
    ["grp_cs", "cs"],
    ["grp_stat", "stat"],
    ["grp_eess", "eess"],
]


def normalise(category_id):
    for aliases in ALIASES:
        if category_id in aliases:
            return aliases[0]
    return category_id
