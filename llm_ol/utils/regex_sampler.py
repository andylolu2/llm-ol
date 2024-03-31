import math
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    DefaultDict,
    Dict,
    FrozenSet,
    List,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import interegular
import interegular.fsm
import torch
from interegular.fsm import (
    FSM,
    Alphabet,
    State,
    TransitionKey,
    _AnythingElseCls,
    anything_else,
)
from outlines.fsm.regex import create_fsm_index_tokenizer, make_deterministic_fsm
from transformers import SPIECE_UNDERLINE, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer
    from vllm import LLM

TransitionTrie = Dict[TransitionKey, "Union[TransitionTrie, State, None]"]


def byte_symbol(byte: int) -> str:
    return f"{byte:02X}" if byte >= 0x80 else chr(byte)


def add_to_transition_trie(
    trie: TransitionTrie,
    key_seq: Sequence[TransitionKey],
    value: Union[State, None],
):
    for key in key_seq[:-1]:
        trie = cast(TransitionTrie, trie.setdefault(key, {}))
        assert isinstance(trie, dict), "key sequence of incompatible length"
    trie[key_seq[-1]] = value


def transition_trie_setdefault(
    trie: TransitionTrie,
    default_trie: TransitionTrie,
):
    for key, default_value in default_trie.items():
        dest_value = trie.get(key)
        if isinstance(dest_value, dict) and isinstance(default_value, dict):
            transition_trie_setdefault(dest_value, default_value)
        elif key not in trie:
            trie[key] = default_value


def make_byte_level_fsm(fsm: FSM, keep_utf8=False) -> FSM:
    """Convert an FSM to a byte-level FSM, expanding multi-byte characters as
    sequences of single-byte transitions. If keep_utf8 is set, the original
    utf-8 characters are kept in the alphabet.
    NOTE: we're representing bytes as strings to keep it type-compatible.
    """

    anything_else_key = fsm.alphabet[anything_else]
    symbol_mapping: Dict[Union[str, _AnythingElseCls], TransitionKey] = {}
    map: Dict[State, Dict[TransitionKey, State]] = {}
    states: List[State] = list(fsm.states)

    # identify all multi-byte characters in the alphabet and build a mapping
    # from the original transition keys to sequences of new keys for each byte
    key_to_key_seqs: Dict[TransitionKey, Set[Tuple[TransitionKey, ...]]] = {}
    all_key_seqs: Set[Tuple[TransitionKey, ...]] = set()
    all_bytes: Set[int] = set()
    max_key = max(fsm.alphabet.values())
    for symbol, transition_key in fsm.alphabet.items():
        assert symbol == anything_else or len(symbol) == 1
        if symbol == anything_else or ord(symbol) < 0x80:
            symbol_mapping[symbol] = transition_key
        else:
            if keep_utf8:
                symbol_mapping[symbol] = transition_key
            key_list: List[TransitionKey] = []
            for byte in symbol.encode("utf-8"):
                symbol = byte_symbol(byte)
                if symbol not in symbol_mapping:
                    symbol_mapping[symbol] = max_key = TransitionKey(max_key + 1)
                    all_bytes.add(byte)
                key_list.append(symbol_mapping[symbol])
            key_seq = tuple(key_list)
            key_to_key_seqs.setdefault(transition_key, set()).add(key_seq)
            all_key_seqs.add(key_seq)

    # add all remaining multi-byte utf-8 bytes to the alphabet
    # (this is required to represent `anything_else`)
    utf8_ranges = {
        1: (0x80, 0xC0),  # continuation bytes
        2: (0xC0, 0xE0),  # 2-byte sequences
        3: (0xE0, 0xF0),  # 3-byte sequences
        4: (0xF0, 0xF8),  # 4-byte sequences
    }
    utf8_all_keys: Dict[int, Set[TransitionKey]] = {
        n: set() for n in utf8_ranges.keys()
    }
    for n, (start, end) in utf8_ranges.items():
        range_key = max_key = TransitionKey(max_key + 1)
        for byte in range(start, end):
            byte_key = symbol_mapping.setdefault(byte_symbol(byte), range_key)
            utf8_all_keys[n].add(byte_key)

    # cache of intermediate transition states by transitions from that state
    state_cache: Dict[FrozenSet[Tuple[TransitionKey, State]], State] = {}

    # helper function to create multi-step transitions between states
    max_state = max(fsm.states)

    def create_seq_transitions(
        seq_transitions_trie: TransitionTrie,
    ) -> Dict[TransitionKey, State]:
        nonlocal max_state
        result: Dict[TransitionKey, State] = {}

        for next_key, next_trie in seq_transitions_trie.items():
            if isinstance(next_trie, dict):
                next_transitions = create_seq_transitions(next_trie)
                if not next_transitions:
                    continue
                cache_key = frozenset(next_transitions.items())
                next_state = state_cache.get(cache_key)
                if next_state is None:
                    next_state = max_state = State(max_state + 1)
                    map[next_state] = next_transitions
                    state_cache[cache_key] = next_state
                    states.append(next_state)
                result[next_key] = next_state
            elif next_trie is not None:
                result[next_key] = next_trie

        return result

    # create new states and transitions
    for state, transitions in fsm.map.items():
        seq_transitions_trie: TransitionTrie = {}
        state_map: Dict[TransitionKey, State] = {}

        for transition_key, to_state in transitions.items():
            if transition_key in key_to_key_seqs:
                if keep_utf8:
                    state_map[transition_key] = to_state
                for key_seq in key_to_key_seqs[transition_key]:
                    add_to_transition_trie(seq_transitions_trie, key_seq, to_state)
            else:  # keep single-byte transitions as is
                state_map[transition_key] = to_state

        # handle multi-byte anything_else sequences
        if anything_else_key in transitions:
            for key_seq in all_key_seqs:
                add_to_transition_trie(seq_transitions_trie, key_seq, None)

            anything_else_trie: TransitionTrie = {}
            cont_trie: Union[TransitionTrie, State] = transitions[anything_else_key]
            for n in range(2, 5):
                cont_trie = {key: cont_trie for key in utf8_all_keys[1]}
                for key in utf8_all_keys[n]:
                    anything_else_trie[key] = cont_trie

            transition_trie_setdefault(seq_transitions_trie, anything_else_trie)

        # create new states and transitions
        next_transitions = create_seq_transitions(seq_transitions_trie)
        state_map.update(next_transitions)
        map[state] = state_map

    return FSM(
        alphabet=Alphabet(symbol_mapping),
        states=states,
        initial=fsm.initial,
        finals=fsm.finals,
        map=map,
    )


def adapt_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Adapt a tokenizer to use to compile the FSM.

    The API of Outlines tokenizers is slightly different to that of `transformers`. In
    addition we need to handle the missing spaces to Llama's tokenizer to be able to
    compile FSMs for this model.

    Parameters
    ----------
    tokenizer
        The tokenizer of the model.

    Returns
    -------
    PreTrainedTokenizerBase
        The adapted tokenizer.
    """
    tokenizer.vocabulary = tokenizer.get_vocab()
    tokenizer.special_tokens = set(tokenizer.all_special_tokens)

    def convert_token_to_string(token: Union[str, bytes]) -> str:
        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if (
            type(token) is str
            and token.startswith(SPIECE_UNDERLINE)
            or token == "<0x20>"
        ):
            return " " + string

        return string

    tokenizer.convert_token_to_string = convert_token_to_string

    return tokenizer


@dataclass(frozen=True)
class Write:
    """Write instruction.

    Attributes
    ----------
    tokens
        The sequence of tokens to be added to the current sequence by the
        generation process.

    """

    tokens: List[int]


@dataclass(frozen=True)
class Generate:
    """Generate instruction

    Attributes
    ----------
    tokens
        The tokens that lead to a valid completion if generated.
    """

    tokens: List[int]


Instruction = Union[Write, Generate]


class Guide(Protocol):
    """Base definition of a generation guide.

    A generation guide defines the behavior of a finite-state machine that guides
    a text generation procedure. Unlike the DFAs built from regular expressions
    guides can also emit a `Write` instructions which tells the model that it can
    append a sequence of tokens (or token word) instead of generating it.

    """

    def get_next_instruction(self, state: int) -> Instruction: ...

    def get_next_state(self, state: int, token_id: int) -> int: ...

    def is_final_state(self, state: int) -> bool: ...

    def copy(self) -> "Guide": ...


class RegexGuide(Guide):
    """Guide to generate text in the language of a regular expression."""

    initial_state = 0

    def __init__(self, regex_string: str, tokenizer):
        def create_states_mapping(
            regex_string: str, cacheable_vocabulary: Tuple[Tuple[str, int], ...]
        ) -> Tuple[dict, set, set]:
            """Create the variables related to the mapping between states and tokens
            The parameters of the function are used for caching purpose
            """
            regex_pattern = interegular.parse_pattern(regex_string)
            byte_fsm = make_byte_level_fsm(
                regex_pattern.to_fsm().reduce(), keep_utf8=True
            )
            regex_fsm, _ = make_deterministic_fsm(byte_fsm)
            states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
                regex_fsm, tokenizer
            )

            # We make sure that it is possible to generate strings in the language
            # of the regular expression with the tokens present in the model's
            # vocabulary.
            if not any(
                regex_fsm.finals.intersection(v.values())
                for v in states_to_token_maps.values()
            ):
                raise ValueError(
                    "The vocabulary does not allow us to build a sequence that matches the input regex"
                )

            return states_to_token_maps, empty_token_ids, regex_fsm.finals

        (
            self.states_to_token_maps,
            self.empty_token_ids,
            fsm_finals,
        ) = create_states_mapping(
            regex_string, tuple(sorted(tokenizer.vocabulary.items()))
        )
        self.vocabulary = list(tokenizer.vocabulary.values())
        self.eos_token_id = tokenizer.eos_token_id
        self.final_states = fsm_finals | {-1}

    def get_next_instruction(self, state: int) -> Instruction:
        """Return the next instruction for guided generation.

        The initialization of the guide builds an index which maps FSM states to a
        map from authorized tokens to the state in which the guide needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the guide. We only authorize EOS tokens in the final
        state.

        Parameters
        ----------
        state
            The current state of the guide.

        Returns
        -------
        A `Generate` instance that contains the model and the allowed token ids.

        """
        next_tokens_to_end_states = self.states_to_token_maps.get(state)
        if next_tokens_to_end_states is None:
            return Write([self.eos_token_id])

        return Generate(list(next_tokens_to_end_states.keys()))

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide.

        We use the index to determine to which state the guide should transition
        given the token that was just generated.

        Parameters
        ----------
        state
            The current state of the guide.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the guide.

        """
        if token_id == self.eos_token_id:
            return -1
        elif (
            state in self.final_states
        ):  # Necessary because we keep generating EOS tokens when finished
            return state

        last_token_to_end_state = self.states_to_token_maps[state]
        next_state = last_token_to_end_state.get(token_id)
        if next_state is None:
            next_state = -1

        return next_state

    @classmethod
    def from_interegular_fsm(
        cls, interegular_fsm: interegular.fsm.FSM, tokenizer: "Tokenizer"
    ):
        from_interegular_instance = cls.__new__(cls)

        def create_states_mapping_from_interegular_fsm(
            fsm: interegular.fsm.FSM, cacheable_vocabulary: Tuple[Tuple[str, int], ...]
        ) -> Tuple[dict, set]:
            """Create the variables related to the mapping between states and tokens
            The parameters of the function are used for caching purpose
            """
            byte_fsm = make_byte_level_fsm(fsm.reduce(), keep_utf8=True)
            regex_fsm, _ = make_deterministic_fsm(byte_fsm)
            states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
                regex_fsm, tokenizer
            )

            # We make sure that it is possible to generate strings in the language
            # of the regular expression with the tokens present in the model's
            # vocabulary.
            if not any(
                regex_fsm.finals.intersection(v.values())
                for v in states_to_token_maps.values()
            ):
                raise ValueError(
                    "The vocabulary does not allow us to build a sequence that matches the input regex"
                )

            return states_to_token_maps, empty_token_ids

        (
            from_interegular_instance.states_to_token_maps,
            from_interegular_instance.empty_token_ids,
        ) = create_states_mapping_from_interegular_fsm(
            interegular_fsm, tuple(sorted(tokenizer.vocabulary.items()))
        )
        from_interegular_instance.vocabulary = list(tokenizer.vocabulary.values())
        from_interegular_instance.eos_token_id = tokenizer.eos_token_id
        return from_interegular_instance

    def is_final_state(self, state: int) -> bool:
        """Determine whether the current state of the guide is a final state."""
        return state in self.final_states

    def copy(self):
        return self


class RegexLogitsProcessor:
    """Bias vLLM generation based on a regular expression.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, regex_string: str, llm: "LLM"):
        """Compile the FSM that drives the regex-structured generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression.
        llm
            The vLLM model.

        Raises
        ------
        ValueError
            If the provided LLM instance in `RegexLogitsProcessor` neither has a
            `tokenizer` attribute or a `get_tokenizer` method.
        """
        tokenizer = llm.get_tokenizer()
        tokenizer = adapt_tokenizer(tokenizer=tokenizer)
        self.fsm = RegexGuide(regex_string, tokenizer)
        self._fsm_state: DefaultDict[int, int] = defaultdict(int)

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token.

        Parameters
        ----------
        input_ids
            The tokens of the current sentence.
        scores
            The logits of the current sentence.

        Returns
        -------
        torch.Tensor
            The biased logits.
        """
        seq_id = hash(tuple(input_ids))

        # Initialize the FSM state dictionary if the input_ids are empty, as this means
        # that the input_ids are the first tokens of the sequence.
        if len(input_ids) > 0:
            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            self._fsm_state[seq_id] = self.fsm.get_next_state(
                state=self._fsm_state[last_seq_id], token_id=last_token
            )

        allowed_tokens = self.fsm.get_next_instruction(
            state=self._fsm_state[seq_id]
        ).tokens

        mask = torch.full((scores.shape[-1],), -math.inf, device=scores.device)
        mask[allowed_tokens] = 0
        biased_scores = scores + mask

        return biased_scores
