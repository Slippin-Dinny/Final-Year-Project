"""
call_screen_openai_tts.py 

End-to-end prototype demostration:
1. Send a mock scam call transcript to OpenAI (gpt-5.1 Responses API)
2. The model returns JSON (classification, reasoning, etc.)
3. Extract 'spoken_response_to_caller'
4. Use OpenAI TTS (gpt-4o-mini-tts) to generate audio
5. Play the MP3 using the system's default audio player
"""

from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import json
import os
import platform
import subprocess

from openai import OpenAI


###### OPENAI CLIENT + API KEY 

# !!!!! Do not commit the real API key to a public github repo. !!!!!
client = OpenAI(api_key="API-KEY-HERE")



###### BASIC INSTRUCTIONS 

BASE_INSTRUCTIONS = """
You are an AI phone call screener protecting an elderly user from scams.

General behaviour:
- The user is a 79 years old man and may be vulnerable to pressure, urgency and confusion.
- You NEVER directly connect the caller to the user. You only *recommend* what to do.
- Always be cautious if the caller asks for money, bank details, passwords, codes,
  remote access to the user's computer/phone, or personal information that is not
  strictly necessary.
- Speak clearly and simply. Avoid technical jargon.

Your outputs will be used to:
1) Decide whether the call should reach the user.
2) Generate a short spoken response back to the caller via Text-To-Speech.
"""



###### SECURITY RULES

SECURITY_RULES: List[str] = [
    "The user is a 79-year-old man. Treat all unexpected calls with extra caution.",
    "If the caller mentions job sites such as Indeed or LinkedIn, treat this as suspicious "
    "unless there is strong evidence that the user is actively expecting that call.",
    "If the caller asks for bank account details, credit/debit card numbers, PINs, "
    "one-time codes, or passwords, classify the call as HIGH RISK.",
    "If the caller pressures the user with urgency (e.g., 'act now', 'you will be arrested', "
    "'your account will be closed today'), this is a strong scam indicator.",
    "If the caller asks the user to install remote access software, classify as HIGH RISK.",
    "If the caller claims to be from a government body, bank, or tech company but cannot "
    "provide clear verifiable information, treat as suspicious.",
]



###### CALL CONTEXT

@dataclass
class CallContext:
    caller_id: Optional[str] = None     # e.g. "+353 83 123 4567" or "Unknown"
    call_type: str = "incoming"         # e.g. "incoming", "missed", "voicemail"
    user_age: int = 79                 



###### PROMPT BUILDER

def build_ai_input(mock_call_text: str, context: CallContext) -> str:
    """
    Build the text that will be sent to the AI as the 'input' prompt.
    """
    rules_text = "\n".join(f"- {rule}" for rule in SECURITY_RULES)

    prompt = f"""
You are screening a phone call for an elderly user.

User profile:
- Age: {context.user_age} years old
- Name: Micheal

Call context:
- Caller ID: {context.caller_id or "Unknown"}
- Call type: {context.call_type}

Security rules you MUST follow:
{rules_text}

Below is the transcript of what the CALLER has said so far.
Treat this as if you are listening to a live phone call.

CALLER TRANSCRIPT:
\"\"\"{mock_call_text}\"\"\"

Your tasks:
1. Classify the call as one of: "safe", "suspicious", or "likely_scam".
2. Briefly explain the main reasons for your classification.
3. List any specific red flags you noticed (if any).
4. Decide what to do next for the user. Choose one:
   - "block_call"
   - "warn_and_block"
   - "allow_through"
   - "ask_more_questions"
5. Write a short, polite sentence that could be spoken back to the caller by a TTS
   system (e.g., "We cannot proceed with this call. Goodbye.").

Respond in **JSON only** with the following keys:
- classification
- reasoning
- red_flags
- action_for_user
- spoken_response_to_caller
"""
    return prompt



###### OPENAI RESPONSES CALL

def send_to_openai(prompt: str) -> str:
    """
    Send the built prompt to OpenAI Responses API and return the raw text output.
    """
    response = client.responses.create(
        model="gpt-5.1",
        instructions=BASE_INSTRUCTIONS,
        input=prompt,
    )
    return response.output_text  



###### OPENAI TTS (gpt-4o-mini-tts)

def speak_with_openai(text: str, filename: str = "call_response.mp3") -> Path:
    """
    Use OpenAI's TTS model to generate speech from text.
    Returns the Path to the saved audio file.
    """
    audio_path = Path(__file__).parent / filename

    # Streaming TTS straight to file (recommended in OpenAI docs)
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        instructions="Speak clearly, calmly, and politely.",
        # response_format defaults to mp3
    ) as response:
        response.stream_to_file(audio_path)

    return audio_path



###### CROSS-PLATFORM AUDIO PLAYBACK

def play_audio(audio_path: Path) -> None:
    """
    Open the given audio file with the OS default media player.
    No extra Python audio libraries required.
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        subprocess.run(["afplay", str(audio_path)], check=False)
    elif system == "Windows":
        # os.startfile is Windows-only
        os.startfile(str(audio_path))  # type: ignore[attr-defined]
    else:  # Linux / other
        # xdg-open will use the default associated player
        subprocess.run(["xdg-open", str(audio_path)], check=False)



###### SCREENING + TTS PIPELINE

def screen_mock_call(mock_call_text: str,
                     caller_id: Optional[str] = None,
                     call_type: str = "incoming"):
    """
    High-level helper:
    - Build context and prompt
    - Call gpt-5.1 via Responses API
    - Parse the JSON
    - Generate TTS with gpt-4o-mini-tts
    - Play the audio reply
    """
    context = CallContext(caller_id=caller_id, call_type=call_type)
    prompt = build_ai_input(mock_call_text, context)
    ai_response_text = send_to_openai(prompt)

    print("\n===== RAW AI RESPONSE TEXT =====")
    print(ai_response_text)
    print("================================\n")

    # Parse JSON (assumes the model followed instructions)
    try:
        result = json.loads(ai_response_text)
    except json.JSONDecodeError:
        print("ERROR: Could not parse JSON from model response.")
        print("Falling back to a generic spoken line.")
        result = {
            "spoken_response_to_caller": "We cannot proceed with this call. Goodbye."
        }

    spoken_line = result.get(
        "spoken_response_to_caller",
        "We cannot proceed with this call. Goodbye."
    )

    print("AI wants to say to the caller:", spoken_line)

    # Generate and play TTS
    audio_path = speak_with_openai(spoken_line)
    print(f"Audio saved to: {audio_path}")
    play_audio(audio_path)



###### MOCK CALL 

if __name__ == "__main__":
    mock_call = """
!!!!! Insert mock call here !!!!!
"""

    screen_mock_call(
        mock_call_text=mock_call,
        caller_id="+353 83 123 4567",
        call_type="incoming",
    )
