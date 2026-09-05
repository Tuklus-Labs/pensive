import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[2] / 'eval/agent_outcomes/score.py'
spec = importlib.util.spec_from_file_location('outcome_score', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ID = '01M1QDTV000000000000000003'


def inputs(evidence):
    case = dict(id='case', category='followup', taskId='task', checkpoints=[],
                dialogue=[{'user': 'We are discussing Birch exports.'}],
                expected={'answer': 'birch export --compat-v2', 'abstained': False})
    packet = dict(id='case', memory=f'p3://{ID} Birch requires birch export --compat-v2.')
    response = dict(id='case', answer='birch export --compat-v2', abstained=False, evidence=evidence)
    return [case], [packet], [response], {'case': [ID]}


def test_context_and_fact_jointly_support_answer():
    report = module.score(*inputs(['recentDialogue', 'p3://' + ID]))
    assert report['correctSupported'] == 1, 'dialogue can resolve the subject while displayed memory supplies the answer'


def test_context_alone_is_not_a_fact():
    report = module.score(*inputs(['recentDialogue']))
    assert report['correctSupported'] == 0, 'topic-only dialogue cannot support a guessed memory fact'


def test_unknown_citation_is_not_support():
    report = module.score(*inputs([ID, 'p3://not-in-this-case']))
    assert report['correctSupported'] == 0, 'a correct answer does not excuse a fabricated or cross-case citation'


@pytest.mark.parametrize('duplicate', [False, True])
def test_unknown_case_set_rejected(duplicate):
    cases, packets, responses, support = inputs([ID])
    responses = responses * 2 if duplicate else []
    with pytest.raises(ValueError, match='exactly one'):
        module.score(cases, packets, responses, support)


def test_reader_schema_rejected():
    cases, packets, responses, support = inputs([ID])
    responses[0]['abstained'] = True
    with pytest.raises(ValueError, match='agree'):
        module.score(cases, packets, responses, support)
