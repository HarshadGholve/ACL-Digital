import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Alarm sound
def play_alarm():
    os.system('play -nq -t alsa synth 0.5 sine 880')

# Email configuration
EMAIL_ADDRESS = "harshad.gholve123@gmail.com"
EMAIL_PASSWORD = "nbji urkp ogfo suzj"
TO_EMAIL = "harshad.gholve21@pccoepune.org"

# Function to send email
def send_email(emotion, threshold):
    try:
        subject = f"Passenger Safety Alert: {emotion.upper()} detected {threshold} times"
        body = f"Dear User,\n\nThe system has detected '{emotion}' emotion {threshold} times.\nPlease check for passenger safety.\n\nBest Regards,\nPassenger Safety System"
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")
