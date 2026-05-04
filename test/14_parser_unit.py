from spych.responders import BaseResponder, AgentResponse
from unittest.mock import MagicMock
import json

def test_parse_output():
    # Mock Spych object for BaseResponder init
    spych_mock = MagicMock()
    responder = BaseResponder(spych_mock)
    
    # Test case 1: Standard JSON
    raw_1 = '{"response": "Hello", "summary": "Hi", "requires_user_feedback": false}'
    resp_1 = responder.parse_output(raw_1)
    assert resp_1.response == "Hello"
    assert resp_1.summary == "Hi"
    assert resp_1.requires_user_feedback is False
    
    # Test case 2: JSON with text around it
    raw_2 = 'Some prefix text {"response": "Hello", "summary": "Hi", "requires_user_feedback": true} some suffix'
    resp_2 = responder.parse_output(raw_2)
    assert resp_2.response == "Hello"
    assert resp_2.requires_user_feedback is True
    
    # Test case 3: JSON with braces in strings (The bug!)
    raw_3 = '{"response": "Content with { braces } inside", "summary": "Braces { }", "requires_user_feedback": false}'
    resp_3 = responder.parse_output(raw_3)
    assert resp_3.response == "Content with { braces } inside"
    assert resp_3.summary == "Braces { }"
    
    # Test case 4: Malformed JSON
    raw_4 = '{"response": "Incomplete'
    resp_4 = responder.parse_output(raw_4)
    assert resp_4.response == raw_4 # Falls back to raw
    
    # Test case 5: Boolean as string
    raw_5 = '{"response": "Test", "summary": "Test", "requires_user_feedback": "true"}'
    resp_5 = responder.parse_output(raw_5)
    assert resp_5.requires_user_feedback is True

    # Test case 6: Intermediate response
    raw_6 = '{"response": "Wait", "summary": "Wait", "requires_user_feedback": false, "is_intermediate_response": true}'
    resp_6 = responder.parse_output(raw_6)
    assert resp_6.is_intermediate_response is True

    print("All parser tests passed!")

if __name__ == "__main__":
    test_parse_output()
