

import os

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")

_TWILIO_CONFIGURED = all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER])


def build_call_message(incident, photo_path=None):
    
    severity = incident["severity"]
    classes = " and ".join(incident["object_classes"])
    hospital = incident["recommended_hospital"]
    return (
        f"This is an automated alert from the Veganodi collision intelligence system. "
        f"A {severity} severity incident involving a {classes} has been detected. "
        f"The recommended hospital is {hospital['name']}, "
        f"estimated {hospital['eta_min']} minutes away. "
        f"Please dispatch emergency response to the reported location immediately."
    )


def place_emergency_call(to_phone_number, incident):
    
    message = build_call_message(incident)

    if not _TWILIO_CONFIGURED:
        print("    ---- EMERGENCY CALL (DRY RUN — Twilio not configured) ----")
        print(f"    Would call: {to_phone_number}")
        print(f"    Would say: \"{message}\"")
        print("    (Set TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM_NUMBER")
        print("     environment variables to place a real call.)")
        print("    ------------------------------------------------------------\n")
        return {"mode": "dry_run", "to": to_phone_number, "message": message}

    try:
        from twilio.rest import Client
        from twilio.twiml.voice_response import VoiceResponse

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        twiml = VoiceResponse()
        twiml.say(message, voice="Polly.Joanna")  
        call = client.calls.create(
            to=to_phone_number,
            from_=TWILIO_FROM_NUMBER,
            twiml=str(twiml),
        )
        print(f"    ---- EMERGENCY CALL PLACED ---- (Twilio SID: {call.sid})")
        print(f"    Called: {to_phone_number}")
        print(f"    Message: \"{message}\"\n")
        return {"mode": "live", "to": to_phone_number, "message": message, "call_sid": call.sid}

    except Exception as e:
      
        print(f"    ---- EMERGENCY CALL FAILED: {e} ----")
        print(f"    (Would have called {to_phone_number} and said: \"{message}\")\n")
        return {"mode": "error", "to": to_phone_number, "message": message, "error": str(e)}



if __name__ == "__main__":
    fake_incident = {
        "severity": "critical",
        "object_classes": ["person", "car"],
        "recommended_hospital": {"name": "BHEL Hospital, Tiruchirappalli", "eta_min": 3.2},
    }
    result = place_emergency_call("+919000011111", fake_incident)
    print("Result:", result)



