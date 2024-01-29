from itertools import product
from typing import Iterator

import regex as re

raw_patterns = [
    r"(?P<src>NP_[\w'-]+),? such as ((?P<tgt>NP_[\w'-]+)( |,|and|or)*)+",
    r"(?P<src>NP_[\w'-]+),? as ((?P<tgt>NP_[\w'-]+)( |,|and|or)*)+",
    r"((?P<tgt>NP_[\w'-]+)( |,|and|or)*)+(and|or) other (?P<src>NP_[\w'-]+)",
    r"(?P<src>NP_[\w'-]+),? include ((?P<tgt>NP_[\w'-]+)( |,|and|or)*)+",
    r"(?P<src>NP_[\w'-]+),? especially ((?P<tgt>NP_[\w'-]+)( |,|and|or)*)+",
]
patterns = [re.compile(p) for p in raw_patterns]


def find_hyponyms(text: str) -> Iterator[tuple[str, str]]:
    for pattern in patterns:
        for m in pattern.finditer(text):
            yield from product(m.captures("src"), m.captures("tgt"))


# --- Other possible patterns ---

# r"(NP_\w+,? such as (NP_\w+ ?(, )?(and |or )?)+)",
# r"(such NP_\w+,? as (NP_\w+ ?(, )?(and |or )?)+)",
# r"((NP_\w+ ?(, )?)+(and |or )?other NP_\w+)",
# r"(NP_\w+,? include (NP_\w+ ?(, )?(and |or )?)+)",
# r"(NP_\w+,? especially (NP_\w+ ?(, )?(and |or )?)+)",

# r"((NP_\w+ ?(, )?)+(and |or )?any other NP_\w+)",
# r"((NP_\w+ ?(, )?)+(and |or )?some other NP_\w+)",
# r"((NP_\w+ ?(, )?)+(and |or )?be a NP_\w+)",
# r"(NP_\w+,? like (NP_\w+ ?,? (and |or )?)*NP_\w)",
# r"such (NP_\w+,? as (NP_\w+ ?,? (and |or )?)+)",
# r"((NP_\w+ ?(, )?)+(and |or )?like other NP_\w+)",
# r"((NP_\w+ ?(, )?)+(and |or )?one of the NP_\w+)",
# r"((NP_\w+ ?(, )?)+(and |or )?one of these NP_\w+)",
# r"((NP_\w+ ?(, )?)+(and |or )?one of those NP_\w+)",
# r"example of (NP_\w+,? be (NP_\w+ ?,? (and |or )?)+)",
# r"((NP_\w+ ?(, )?)+(and |or )?be example of NP_\w+)",
# r"(NP_\w+,? for example,? (NP_\w+ ?(, )?(and |or )?)+)",
# r"((NP_\w+ ?(, )?)+(and |or )?which be call NP_\w+)",
# r"((NP_\w+ ?(, )?)+(and |or )?which be name NP_\w+)",
# r"(NP_\w+,? mainly (NP_\w+ ?,? (and |or )?)+)",
# r"(NP_\w+,? mostly (NP_\w+ ?,? (and |or )?)+)",
# r"(NP_\w+,? notably (NP_\w+ ?,? (and |or )?)+)",
# r"(NP_\w+,? particularly (NP_\w+ ?,? (and |or )?)+)",
# r"(NP_\w+,? principally (NP_\w+ ?,? (and |or )?)+)",
# r"(NP_\w+,? in particular (NP_\w+ ?,? (and |or )?)+)",
# r"(NP_\w+,? except (NP_\w+ ?,? (and |or )?)+)",
# r"(NP_\w+,? other than (NP_\w+ ?,? (and |or )?)+)",
# r"(NP_\w+,? e.g.,? (NP_\w+ ?,? (and |or )?)+)",
# r"(NP_\w+ \( (e.g.|i.e.),? (NP_\w+ ?,? (and |or )?)+(\. )?\))",
# r"(NP_\w+,? i.e.,? (NP_\w+ ?,? (and |or )?)+)",
# r"((NP_\w+ ?(, )?)+(and|or)? a kind of NP_\w+)",
# r"((NP_\w+ ?(, )?)+(and|or)? kind of NP_\w+)",
# r"((NP_\w+ ?(, )?)+(and|or)? form of NP_\w+)",
# r"((NP_\w+ ?(, )?)+(and |or )?which look like NP_\w+)",
# r"((NP_\w+ ?(, )?)+(and |or )?which sound like NP_\w+)",
# r"(NP_\w+,? which be similar to (NP_\w+ ?,? (and |or )?)+)",
# r"(NP_\w+,? example of this be (NP_\w+ ?,? (and |or )?)+)",
# r"(NP_\w+,? type (NP_\w+ ?,? (and |or )?)+)",
# r"((NP_\w+ ?(, )?)+(and |or )? NP_\w+ type)",
# r"(NP_\w+,? whether (NP_\w+ ?,? (and |or )?)+)",
# r"(compare (NP_\w+ ?(, )?)+(and |or )?with NP_\w+)",
# r"(NP_\w+,? compare to (NP_\w+ ?,? (and |or )?)+)",
# # r"(NP_\w+,? among -PRON- (NP_\w+ ?,? (and |or )?)+)",
# # r"((NP_\w+ ?(, )?)+(and |or )?as NP_\w+)",  <-- bad
# r"(NP_\w+,?  (NP_\w+ ?,? (and |or )?)+ for instance)",
# r"((NP_\w+ ?(, )?)+(and|or)? sort of NP_\w+)",
# r"(NP_\w+,? which may include (NP_\w+ ?(, )?(and |or )?)+)",
