"""Tests for episodic.kg.validator."""

import pytest

from episodic.kg.prompt_template import normalize_text
from episodic.kg.trigger_tokens import check_trigger
from episodic.kg.validator import (
    validate_patch, VALIDATOR_VERSION,
    STRIP_EDGE_REJECTION_CUE, STRIP_EDGE_NEGATE_POLARITY,
    STRIP_EDGE_DOMAIN_RANGE_VIOLATION, STRIP_EDGE_MISSING_OBJECT_MENTION,
    STRIP_EDGE_USER_SELF_AS_OBJECT, STRIP_EDGE_BASIC_VALIDATION,
    STRIP_ASSERTION_INVALID, STRIP_ENTITY_INVALID,
)


SOURCE_TEXT = "I use Vim daily and I prefer Python over Java."


def _minimal_patch(node_id=1, **overrides):
    """Build a minimal valid patch for testing."""
    patch = {
        'schema_version': 'kg_patch_v1',
        'node_id': node_id,
        'assertions': [{
            'assertion_key': 'a1',
            'span_start': 0,
            'span_end': 18,
            'asserted_by': 'user',
            'polarity': 'affirm',
            'certainty': 'explicit',
            'status': 'active',
            'tags': [],
        }],
        'entities': [],
        'aliases': [],
        'mentions': [{
            'mention_key': 'm1',
            'span_start': 6,
            'span_end': 9,
            'surface_text': 'Vim',
            'entity_ref': None,
            'confidence': 0.9,
            'source_assertion': 'a1',
        }],
        'edges': [],
        'notes': None,
    }
    patch.update(overrides)
    return patch


def test_valid_minimal_patch():
    """Patch with one assertion, one mention, no edges passes validation."""
    patch = _minimal_patch()
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert result.errors == []
    assert result.cleaned_patch is not None


def test_invalid_schema_version():
    """Wrong schema_version is a fatal error."""
    patch = _minimal_patch(schema_version='kg_patch_v999')
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert not result.valid
    assert any('schema_version' in e for e in result.errors)


def test_node_id_mismatch():
    """Patch node_id != expected node_id is a fatal error."""
    patch = _minimal_patch(node_id=999)
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert not result.valid
    assert any('mismatch' in e for e in result.errors)


def test_missing_required_keys():
    """Patch missing 'assertions' key is a fatal error."""
    patch = {
        'schema_version': 'kg_patch_v1',
        'node_id': 1,
        'entities': [],
        'aliases': [],
        'mentions': [],
        'edges': [],
    }
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert not result.valid
    assert any('Missing required keys' in e for e in result.errors)


def test_span_out_of_bounds():
    """span_end > len(source_text) strips the assertion."""
    patch = _minimal_patch()
    patch['assertions'][0]['span_end'] = 9999
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['assertions']) == 0
    assert any('span_end' in w for w in result.warnings)


def test_span_start_negative():
    """span_start < 0 strips the assertion."""
    patch = _minimal_patch()
    patch['assertions'][0]['span_start'] = -1
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['assertions']) == 0
    assert any('span_start' in w for w in result.warnings)


def test_invalid_entity_type():
    """entity_type='project' strips the entity."""
    patch = _minimal_patch()
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'project',
        'canonical_name': 'Test',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['entities']) == 0
    assert any('entity_type' in w for w in result.warnings)


def test_invalid_predicate():
    """predicate='located_in' strips the edge."""
    patch = _minimal_patch()
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'located_in',
        'obj_ref': 'user:self',
        'source_assertion': 'a1',
        'confidence': 0.9,
    }]
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 0
    assert any('predicate' in w for w in result.warnings)


def test_invalid_polarity():
    """polarity='maybe' strips the assertion."""
    patch = _minimal_patch()
    patch['assertions'][0]['polarity'] = 'maybe'
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['assertions']) == 0
    assert any('polarity' in w for w in result.warnings)


def test_invalid_tag():
    """Unknown tag strips the assertion."""
    patch = _minimal_patch()
    patch['assertions'][0]['tags'] = ['SENTIMENT_POS', 'UNKNOWN_TAG']
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['assertions']) == 0
    assert any('UNKNOWN_TAG' in w for w in result.warnings)


def test_surface_text_mismatch():
    """Mention surface_text != source_text[span_start:span_end] strips mention."""
    patch = _minimal_patch()
    patch['mentions'][0]['surface_text'] = 'WRONG'
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['mentions']) == 0
    assert any('surface_text' in w for w in result.warnings)


def test_trigger_token_present():
    """Edge with 'uses' predicate and 'I use Vim' span passes trigger check."""
    patch = _minimal_patch()
    # Assertion covers "I use Vim daily"
    patch['assertions'][0]['span_start'] = 0
    patch['assertions'][0]['span_end'] = 18
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 6,
        'span_end': 9,
        'surface_text': 'Vim',
        'entity_ref': 'e1',
        'confidence': 0.9,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'uses',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.95,
    }]
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid, f"Errors: {result.errors}"
    assert len(result.cleaned_patch['edges']) == 1


def test_no_trigger_token_still_accepted():
    """Edge with 'uses' predicate is accepted even without trigger language.

    Trigger token checking was removed — the LLM understands predicate
    semantics better than any hard-coded word list.
    """
    text = "Vim is a text editor."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 21,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 0,
        'span_end': 3,
        'surface_text': 'Vim',
        'entity_ref': 'e1',
        'confidence': 0.9,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'uses',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.9,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 1


def test_trigger_token_case_insensitive():
    """'USING' matches trigger 'using'."""
    text = "I AM USING Vim for coding."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 25,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 11,
        'span_end': 14,
        'surface_text': 'Vim',
        'entity_ref': 'e1',
        'confidence': 0.9,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'uses',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.9,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid, f"Errors: {result.errors}"
    assert len(result.cleaned_patch['edges']) == 1


def test_cross_topic_resolution_rejected():
    """Entity ref db:<id> out of topic scope without canonical_key is stripped."""
    patch = _minimal_patch()
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Widget',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': {
            'kind': 'map_to_existing',
            'candidate_entity_id': 42,
            'confidence': 0.8,
        },
    }]
    # topic_entity_ids doesn't include 42, and no canonical_key for 42
    result = validate_patch(patch, SOURCE_TEXT, 1, {1, 2, 3}, {})
    assert result.valid
    assert len(result.cleaned_patch['entities']) == 0
    assert any('cross-topic' in w for w in result.warnings)


def test_global_canonical_key_resolution_allowed():
    """Entity ref db:<id> where entity has canonical_key is allowed."""
    patch = _minimal_patch()
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Widget',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': {
            'kind': 'map_to_existing',
            'candidate_entity_id': 42,
            'confidence': 0.8,
        },
    }]
    # 42 has a canonical_key in the global map
    result = validate_patch(
        patch, SOURCE_TEXT, 1, {1, 2, 3},
        {'widget:global': 42}
    )
    # Should not have cross-topic warning
    cross_topic_warnings = [w for w in result.warnings if 'cross-topic' in w]
    assert len(cross_topic_warnings) == 0
    assert len(result.cleaned_patch['entities']) == 1


def test_self_loop_rejected():
    """Edge where subj_ref == obj_ref is stripped."""
    patch = _minimal_patch()
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'uses',
        'obj_ref': 'user:self',
        'source_assertion': 'a1',
        'confidence': 0.9,
    }]
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 0
    assert any('self-loop' in w for w in result.warnings)


def test_no_orphan_entity_warning():
    """Entity with no mentions or edges produces warning."""
    patch = _minimal_patch()
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Orphan',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    # Orphan is a warning, not an error
    assert result.valid
    assert any('orphan' in w.lower() for w in result.warnings)


def test_user_self_reference():
    """subj_ref='user:self' resolves correctly."""
    patch = _minimal_patch()
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 6,
        'span_end': 9,
        'surface_text': 'Vim',
        'entity_ref': 'e1',
        'confidence': 0.9,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'uses',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.9,
    }]
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid, f"Errors: {result.errors}"


def test_db_reference():
    """entity_ref='db:42' resolves when entity_id=42 exists."""
    patch = _minimal_patch()
    patch['mentions'][0]['entity_ref'] = 'db:42'
    result = validate_patch(
        patch, SOURCE_TEXT, 1, {42}, {'key:test': 42}
    )
    assert result.valid, f"Errors: {result.errors}"


def test_db_reference_nonexistent():
    """entity_ref='db:bad' strips the mention."""
    patch = _minimal_patch()
    patch['mentions'][0]['entity_ref'] = 'db:bad'
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['mentions']) == 0
    assert any('unresolvable' in w for w in result.warnings)


def test_invalid_assertion_key_format():
    """assertion_key must match /^a\\d+$/ — invalid key strips assertion."""
    patch = _minimal_patch()
    patch['assertions'][0]['assertion_key'] = 'bad_key'
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['assertions']) == 0
    assert any('assertion_key' in w for w in result.warnings)


def test_invalid_mention_key_format():
    """mention_key must match /^m\\d+$/ — invalid key strips mention."""
    patch = _minimal_patch()
    patch['mentions'][0]['mention_key'] = 'x1'
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['mentions']) == 0


def test_confidence_out_of_range():
    """confidence > 1 strips the mention."""
    patch = _minimal_patch()
    patch['mentions'][0]['confidence'] = 1.5
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['mentions']) == 0
    assert any('confidence' in w for w in result.warnings)


def test_asserted_by_not_user():
    """asserted_by='assistant' strips the assertion in Phase 0."""
    patch = _minimal_patch()
    patch['assertions'][0]['asserted_by'] = 'assistant'
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['assertions']) == 0
    assert any('asserted_by' in w for w in result.warnings)


def test_cascade_assertion_removal_strips_dependents():
    """Removing an assertion cascades to entities, mentions, edges."""
    patch = _minimal_patch()
    # Bad assertion → everything that references a1 should also be stripped
    patch['assertions'][0]['polarity'] = 'invalid_value'
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'uses',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.9,
    }]
    result = validate_patch(patch, SOURCE_TEXT, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['assertions']) == 0
    assert len(result.cleaned_patch['entities']) == 0
    assert len(result.cleaned_patch['mentions']) == 0
    assert len(result.cleaned_patch['edges']) == 0


def test_partial_strip_keeps_good_elements():
    """Bad edge is stripped but good assertion, entity, mention survive."""
    text = "I use Vim daily and I prefer Python over Java."
    patch = _minimal_patch()
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 6,
        'span_end': 9,
        'surface_text': 'Vim',
        'entity_ref': 'e1',
        'confidence': 0.9,
        'source_assertion': 'a1',
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['edges'] = [
        {
            'subj_ref': 'user:self',
            'predicate': 'uses',
            'obj_ref': 'e1',
            'source_assertion': 'a1',
            'confidence': 0.95,
        },
        {
            'subj_ref': 'user:self',
            'predicate': 'located_in',  # bad predicate
            'obj_ref': 'e1',
            'source_assertion': 'a1',
            'confidence': 0.9,
        },
    ]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['assertions']) == 1
    assert len(result.cleaned_patch['entities']) == 1
    assert len(result.cleaned_patch['edges']) == 1
    assert result.cleaned_patch['edges'][0]['predicate'] == 'uses'


# --- Smart quote normalization tests ---


class TestNormalizeText:
    """Tests for normalize_text() smart quote replacement."""

    def test_double_curly_quotes(self):
        assert normalize_text('\u201cHello\u201d') == '"Hello"'

    def test_single_curly_quotes(self):
        assert normalize_text('\u2018world\u2019') == "'world'"

    def test_em_dash(self):
        assert normalize_text('foo\u2014bar') == 'foo--bar'

    def test_en_dash(self):
        assert normalize_text('2020\u20132025') == '2020-2025'

    def test_ellipsis(self):
        assert normalize_text('wait\u2026') == 'wait...'

    def test_mixed(self):
        raw = '\u201cI\u2019m using Python\u2014it\u2019s great\u201d'
        expected = '"I\'m using Python--it\'s great"'
        assert normalize_text(raw) == expected

    def test_ascii_unchanged(self):
        text = 'plain ASCII text with "quotes" and \'apostrophes\''
        assert normalize_text(text) == text

    def test_empty_string(self):
        assert normalize_text('') == ''


def test_smart_quote_validation_passes():
    """Validator normalizes source_text, so spans computed against ASCII pass."""
    # Source text has smart quotes — validator should normalize before checking
    smart_source = '\u201cI use Vim daily and I prefer Python over Java.\u201d'
    # After normalization: '"I use Vim daily and I prefer Python over Java."'
    normalized = normalize_text(smart_source)

    # Build patch with spans against the normalized text
    vim_start = normalized.index('Vim')
    vim_end = vim_start + 3

    patch = {
        'schema_version': 'kg_patch_v1',
        'node_id': 1,
        'assertions': [{
            'assertion_key': 'a1',
            'span_start': 1,  # after opening quote
            'span_end': 19,   # "I use Vim daily an"
            'asserted_by': 'user',
            'polarity': 'affirm',
            'certainty': 'explicit',
            'status': 'active',
            'tags': [],
        }],
        'entities': [],
        'aliases': [],
        'mentions': [{
            'mention_key': 'm1',
            'span_start': vim_start,
            'span_end': vim_end,
            'surface_text': 'Vim',
            'entity_ref': None,
            'confidence': 0.9,
            'source_assertion': 'a1',
        }],
        'edges': [],
        'notes': None,
    }

    # Pass the SMART (un-normalized) source — validator should normalize it
    result = validate_patch(patch, smart_source, 1, set(), {})
    assert result.valid, f"Errors: {result.errors}"


def test_smart_quote_span_length_change():
    """Normalization changes string length for multi-char replacements."""
    # em dash (1 char) → '--' (2 chars), ellipsis (1 char) → '...' (3 chars)
    text = 'A\u2014B\u2026C'
    normalized = normalize_text(text)
    # 'A' + '--' + 'B' + '...' + 'C' = 'A--B...C'
    assert normalized == 'A--B...C'
    assert len(normalized) == 8  # was 5 chars


# --- 'has' predicate tests ---


class TestHasTriggerTokens:
    """Tests for 'has' trigger token matching."""

    def test_have(self):
        assert check_trigger('has', 'I have a dog')

    def test_my(self):
        assert check_trigger('has', 'my cat is named Luna')

    def test_owns(self):
        assert check_trigger('has', 'She owns a Tesla')

    def test_got(self):
        assert check_trigger('has', 'I got a new guitar')

    def test_ive_got(self):
        assert check_trigger('has', "I've got a duplex microphone")

    def test_no_trigger(self):
        assert not check_trigger('has', 'The sky is blue')


def test_has_predicate_validates():
    """Edge with 'has' predicate and 'I have' in span passes."""
    text = "I have a golden retriever named Max."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 35,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Max',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 32,
        'span_end': 35,
        'surface_text': 'Max',
        'entity_ref': 'e1',
        'confidence': 0.9,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'has',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.95,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid, f"Errors: {result.errors}"
    assert len(result.cleaned_patch['edges']) == 1


def test_has_no_trigger_still_accepted():
    """Edge with 'has' predicate accepted even without trigger language.

    Trigger token checking was removed — the LLM understands predicate
    semantics better than any hard-coded word list.
    """
    text = "Python is a programming language."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 32,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Python',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 0,
        'span_end': 6,
        'surface_text': 'Python',
        'entity_ref': 'e1',
        'confidence': 0.9,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'has',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.9,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 1


# --- Check 9h/9i/9j tests ---


def test_user_self_as_object_rejected():
    """Edge with user:self as obj_ref (non-related_to) is stripped."""
    text = "Biscuit is a golden retriever."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 30,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_name': 'Biscuit',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 0,
        'span_end': 7,
        'surface_text': 'Biscuit',
        'entity_ref': 'e1',
        'confidence': 0.9,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'e1',
        'predicate': 'is_a',
        'obj_ref': 'user:self',
        'source_assertion': 'a1',
        'confidence': 0.9,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 0
    assert any(STRIP_EDGE_USER_SELF_AS_OBJECT in w for w in result.warnings)


def test_related_to_allows_user_self_object():
    """related_to predicate allows user:self as obj_ref."""
    text = "Emma is my daughter."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 20,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_name': 'Emma',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [
        {
            'mention_key': 'm1',
            'span_start': 0,
            'span_end': 4,
            'surface_text': 'Emma',
            'entity_ref': 'e1',
            'confidence': 0.95,
            'source_assertion': 'a1',
        },
    ]
    patch['edges'] = [{
        'subj_ref': 'e1',
        'predicate': 'related_to',
        'obj_ref': 'user:self',
        'source_assertion': 'a1',
        'confidence': 0.95,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 1


def test_domain_range_violation_rejected():
    """topic related_to topic violates domain/range — stripped."""
    text = "Python is related to JavaScript."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 31,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [
        {
            'entity_key': 'e1',
            'entity_type': 'topic',
            'canonical_name': 'Python',
            'canonical_key': None,
            'created_by_assertion': 'a1',
            'resolution_hint': None,
        },
        {
            'entity_key': 'e2',
            'entity_type': 'topic',
            'canonical_name': 'JavaScript',
            'canonical_key': None,
            'created_by_assertion': 'a1',
            'resolution_hint': None,
        },
    ]
    patch['mentions'] = [
        {
            'mention_key': 'm1',
            'span_start': 0,
            'span_end': 6,
            'surface_text': 'Python',
            'entity_ref': 'e1',
            'confidence': 0.9,
            'source_assertion': 'a1',
        },
        {
            'mention_key': 'm2',
            'span_start': 21,
            'span_end': 31,
            'surface_text': 'JavaScript',
            'entity_ref': 'e2',
            'confidence': 0.9,
            'source_assertion': 'a1',
        },
    ]
    patch['edges'] = [{
        'subj_ref': 'e1',
        'predicate': 'related_to',
        'obj_ref': 'e2',
        'source_assertion': 'a1',
        'confidence': 0.9,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 0
    assert any('edge_domain_range_violation' in w for w in result.warnings)


def test_missing_mentions_rejected():
    """Edge referencing e1 with no mention for e1 and name not in span is stripped."""
    text = "I use tools daily."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 18,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    # No mention with entity_ref='e1', and "Vim" not in span text
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 6,
        'span_end': 11,
        'surface_text': 'tools',
        'entity_ref': None,
        'confidence': 0.9,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'uses',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.95,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 0
    assert any('edge_missing_object_mention' in w for w in result.warnings)


def test_mention_fallback_canonical_name_match():
    """Edge with no mention but canonical_name in span text survives."""
    text = "I use Vim daily."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 16,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    # Mention exists but has entity_ref=None — so no mention for e1
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 6,
        'span_end': 9,
        'surface_text': 'Vim',
        'entity_ref': None,
        'confidence': 0.9,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'uses',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.95,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    # Fallback: "Vim" (canonical_name) appears in span "I use Vim daily."
    assert len(result.cleaned_patch['edges']) == 1


def test_mention_fallback_no_match():
    """Edge with no mention and canonical_name NOT in span is stripped."""
    text = "I use tools daily."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 18,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 6,
        'span_end': 11,
        'surface_text': 'tools',
        'entity_ref': None,
        'confidence': 0.9,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'uses',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.95,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    # "Vim" does NOT appear in "I use tools daily." — edge stripped
    assert len(result.cleaned_patch['edges']) == 0
    assert any('edge_missing_object_mention' in w for w in result.warnings)


def test_correct_edge_passes():
    """Emma located_at MIT with both mentions inside assertion span passes."""
    text = "My daughter Emma studies computer science at MIT."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 49,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [
        {
            'entity_key': 'e1',
            'entity_type': 'person',
            'canonical_name': 'Emma',
            'canonical_key': None,
            'created_by_assertion': 'a1',
            'resolution_hint': None,
        },
        {
            'entity_key': 'e2',
            'entity_type': 'org',
            'canonical_name': 'MIT',
            'canonical_key': None,
            'created_by_assertion': 'a1',
            'resolution_hint': None,
        },
    ]
    patch['mentions'] = [
        {
            'mention_key': 'm1',
            'span_start': 12,
            'span_end': 16,
            'surface_text': 'Emma',
            'entity_ref': 'e1',
            'confidence': 0.95,
            'source_assertion': 'a1',
        },
        {
            'mention_key': 'm2',
            'span_start': 45,
            'span_end': 48,
            'surface_text': 'MIT',
            'entity_ref': 'e2',
            'confidence': 0.95,
            'source_assertion': 'a1',
        },
    ]
    patch['edges'] = [{
        'subj_ref': 'e1',
        'predicate': 'located_at',
        'obj_ref': 'e2',
        'source_assertion': 'a1',
        'confidence': 0.95,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid, f"Errors: {result.errors}, Warnings: {result.warnings}"
    assert len(result.cleaned_patch['edges']) == 1
    assert result.cleaned_patch['edges'][0]['predicate'] == 'located_at'


def test_rejection_cue_strips_edge():
    """Edge in assertion with rejection cue is stripped entirely."""
    text = "Google Brain offered me a position, but academic freedom won out."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 64,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'org',
        'canonical_name': 'Google Brain',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 0,
        'span_end': 12,
        'surface_text': 'Google Brain',
        'entity_ref': 'e1',
        'confidence': 0.95,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'located_at',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.95,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 0
    assert any(STRIP_EDGE_REJECTION_CUE in w for w in result.warnings)


def test_temporal_cue_adds_time_past_tag():
    """Edge in assertion with temporal cue gets TIME_PAST tag added."""
    text = "I used to work at IBM Research before moving to academia."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 56,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'org',
        'canonical_name': 'IBM Research',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 18,
        'span_end': 30,
        'surface_text': 'IBM Research',
        'entity_ref': 'e1',
        'confidence': 0.95,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'located_at',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.95,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    # Edge kept but TIME_PAST tag added
    assert len(result.cleaned_patch['edges']) == 1
    assert any('edge_temporal_cue_added_time_past' in w for w in result.warnings)
    # Verify the assertion now has TIME_PAST tag
    tags = result.cleaned_patch['assertions'][0].get('tags', [])
    assert 'TIME_PAST' in tags


def test_no_cue_edge_unchanged():
    """Edge in assertion without rejection/temporal cues passes unchanged."""
    text = "I work at MIT doing NLP research."
    patch = _minimal_patch()
    patch['assertions'] = [{
        'assertion_key': 'a1',
        'span_start': 0,
        'span_end': 32,
        'asserted_by': 'user',
        'polarity': 'affirm',
        'certainty': 'explicit',
        'status': 'active',
        'tags': [],
    }]
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'org',
        'canonical_name': 'MIT',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'] = [{
        'mention_key': 'm1',
        'span_start': 10,
        'span_end': 13,
        'surface_text': 'MIT',
        'entity_ref': 'e1',
        'confidence': 0.95,
        'source_assertion': 'a1',
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'located_at',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.95,
    }]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 1
    # No rejection or temporal warnings
    assert not any(STRIP_EDGE_REJECTION_CUE in w for w in result.warnings)
    assert not any('edge_temporal_cue_added_time_past' in w for w in result.warnings)


def test_all_strips_have_reason_code():
    """Every strip warning contains a valid reason code, no unclassified."""
    text = "Google Brain offered me a position. I use tools daily."
    patch = _minimal_patch()
    patch['assertions'] = [
        {
            'assertion_key': 'a1',
            'span_start': 0,
            'span_end': 34,
            'asserted_by': 'user',
            'polarity': 'affirm',
            'certainty': 'explicit',
            'status': 'active',
            'tags': [],
        },
        {
            'assertion_key': 'a2',
            'span_start': 35,
            'span_end': 54,
            'asserted_by': 'user',
            'polarity': 'affirm',
            'certainty': 'explicit',
            'status': 'active',
            'tags': [],
        },
    ]
    patch['entities'] = [
        {
            'entity_key': 'e1',
            'entity_type': 'org',
            'canonical_name': 'Google Brain',
            'canonical_key': None,
            'created_by_assertion': 'a1',
            'resolution_hint': None,
        },
        {
            'entity_key': 'e2',
            'entity_type': 'topic',
            'canonical_name': 'Python',
            'canonical_key': None,
            'created_by_assertion': 'a2',
            'resolution_hint': None,
        },
    ]
    patch['mentions'] = [
        {
            'mention_key': 'm1',
            'span_start': 0,
            'span_end': 12,
            'surface_text': 'Google Brain',
            'entity_ref': 'e1',
            'confidence': 0.95,
            'source_assertion': 'a1',
        },
        {
            'mention_key': 'm2',
            'span_start': 41,
            'span_end': 46,
            'surface_text': 'tools',
            'entity_ref': None,
            'confidence': 0.9,
            'source_assertion': 'a2',
        },
    ]
    patch['edges'] = [
        {
            # Rejection cue → stripped
            'subj_ref': 'user:self',
            'predicate': 'located_at',
            'obj_ref': 'e1',
            'source_assertion': 'a1',
            'confidence': 0.95,
        },
        {
            # Missing object mention (Python not in "I use tools daily.") → stripped
            'subj_ref': 'user:self',
            'predicate': 'uses',
            'obj_ref': 'e2',
            'source_assertion': 'a2',
            'confidence': 0.95,
        },
        {
            # Domain/range: topic related_to topic → stripped
            'subj_ref': 'e2',
            'predicate': 'related_to',
            'obj_ref': 'e2',
            'source_assertion': 'a2',
            'confidence': 0.95,
        },
    ]
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid

    # Collect all strip warnings
    strip_warnings = [w for w in result.warnings if w.startswith('stripped:')]
    assert len(strip_warnings) > 0, "Expected some strips"

    # Every strip warning must contain a known reason code
    from episodic.kg.validator import (
        STRIP_ASSERTION_INVALID, STRIP_ENTITY_INVALID,
        STRIP_MENTION_INVALID, STRIP_ALIAS_INVALID,
        STRIP_EDGE_USER_SELF_AS_OBJECT, STRIP_EDGE_DOMAIN_RANGE_VIOLATION,
        STRIP_EDGE_MISSING_SUBJECT_MENTION, STRIP_EDGE_MISSING_OBJECT_MENTION,
        STRIP_EDGE_REJECTION_CUE, STRIP_EDGE_NEGATE_POLARITY,
        STRIP_EDGE_BASIC_VALIDATION,
        STRIP_CASCADE_ASSERTION, STRIP_CASCADE_ENTITY,
    )
    known_reasons = {
        STRIP_ASSERTION_INVALID, STRIP_ENTITY_INVALID,
        STRIP_MENTION_INVALID, STRIP_ALIAS_INVALID,
        STRIP_EDGE_USER_SELF_AS_OBJECT, STRIP_EDGE_DOMAIN_RANGE_VIOLATION,
        STRIP_EDGE_MISSING_SUBJECT_MENTION, STRIP_EDGE_MISSING_OBJECT_MENTION,
        STRIP_EDGE_REJECTION_CUE, STRIP_EDGE_NEGATE_POLARITY,
        STRIP_EDGE_BASIC_VALIDATION,
        STRIP_CASCADE_ASSERTION, STRIP_CASCADE_ENTITY,
    }
    for w in strip_warnings:
        has_reason = any(reason in w for reason in known_reasons)
        assert has_reason, f"Strip without known reason code: {w}"
