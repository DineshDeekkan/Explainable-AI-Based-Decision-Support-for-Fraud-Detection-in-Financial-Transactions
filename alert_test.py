from twilio.rest import Client

ACCOUNT_SID = "ACc0b7031f35ec6f1120d2713793519526"
AUTH_TOKEN = "cdfa79f64c08c5b200383824107de46b"

client = Client(ACCOUNT_SID, AUTH_TOKEN)

FROM_NUMBER = "whatsapp:+14155238886"
TO_NUMBER = "whatsapp:+917695967275"   

try:
    message = client.messages.create(
        body="Test message from fraud system",
        from_=FROM_NUMBER,
        to=TO_NUMBER
    )
    print("✅ Message sent successfully!")
except Exception as e:
    print("❌ Error:", e)