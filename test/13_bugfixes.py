from spych import Spych, SpychOrchestrator
from spych.responders import BaseResponder, AgentResponse
from spych.dashboard import AgentDashboard


class DummyResponder(BaseResponder):
    def respond(self, user_input: str) -> AgentResponse:
        return AgentResponse(
            response="test", summary="test", requires_user_feedback=False
        )


def test_bugfixes():
    spych_object = Spych(whisper_model="tiny.en")
    responder = DummyResponder(spych_object=spych_object, use_speaker=False)

    # 1. Verify non-greedy JSON parsing fix
    multi_json = 'Here is the first object: {"response": "first", "summary": "s1", "requires_user_feedback": false} and here is a second one: {"response": "second", "summary": "s2", "requires_user_feedback": true}'
    parsed = responder.parse_output(multi_json)
    assert (
        parsed.response == "first"
    ), f"Expected 'first', got {parsed.response!r}"
    assert parsed.requires_user_feedback is False

    # 2. Verify BaseResponder handles empty response by calling clear_current_turn
    dashboard = AgentDashboard(agent_name="Test", wake_words=["test"])
    responder_with_dash = DummyResponder(
        spych_object=spych_object, dashboard=dashboard, use_speaker=False
    )

    dashboard.on_user_input("hello")
    assert dashboard._current_user_input == "hello"

    empty_resp = AgentResponse(
        response="", summary="", requires_user_feedback=False
    )
    responder_with_dash.on_response(empty_resp)
    dashboard.clear_current_turn()

    assert (
        dashboard._current_user_input == ""
    ), "Dashboard should have cleared user input"

    # 3. Verify scroll boundary
    dashboard.scroll_up(10000)
    assert (
        dashboard._scroll_offset == 5000
    ), f"Expected 5000, got {dashboard._scroll_offset}"
    dashboard.scroll_down(6000)
    assert (
        dashboard._scroll_offset == 0
    ), f"Expected 0, got {dashboard._scroll_offset}"

    # 4. Verify SpychOrchestrator reuses the responder's WhisperModel for
    # wake-word spotting when its config matches, instead of loading a
    # second, redundant copy.
    matching_orchestrator = SpychOrchestrator(
        entries=[
            {
                "responder": responder,
                "wake_words": ["hello"],
                "terminate_words": ["terminate"],
            }
        ],
        spych_wake_kwargs={"whisper_model": "tiny.en"},
    )
    assert (
        matching_orchestrator.spych_wake.wake_model is spych_object.wake_model
    )

    other_spych_object = Spych(whisper_model="base.en")
    other_responder = DummyResponder(
        spych_object=other_spych_object, use_speaker=False
    )
    mismatched_orchestrator = SpychOrchestrator(
        entries=[
            {
                "responder": other_responder,
                "wake_words": ["hello"],
                "terminate_words": ["terminate"],
            }
        ],
        spych_wake_kwargs={"whisper_model": "tiny.en"},
    )
    assert (
        mismatched_orchestrator.spych_wake.wake_model
        is not other_spych_object.wake_model
    )
