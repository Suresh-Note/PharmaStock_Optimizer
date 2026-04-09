"""
Email Service — SMTP Email Delivery

Handles all outgoing email communications using Gmail SMTP.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import settings
from utils.logger import get_logger
from utils.exceptions import EmailConfigurationError, EmailDeliveryError

logger = get_logger(__name__)


class EmailService:
    """Handles SMTP email delivery."""

    def __init__(self):
        self.from_email = settings.SMTP_EMAIL
        self.from_password = settings.SMTP_PASSWORD
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT

    def _validate_config(self):
        """Ensure SMTP credentials are configured."""
        if not self.from_email or not self.from_password:
            raise EmailConfigurationError()

    def send(self, to_email: str, subject: str, body: str) -> bool:
        """
        Send an email via SMTP.
        
        Args:
            to_email: Recipient email address
            subject: Email subject line
            body: Plain text email body
            
        Returns:
            True if email sent successfully
            
        Raises:
            EmailConfigurationError: If SMTP credentials are missing
            EmailDeliveryError: If sending fails
        """
        self._validate_config()

        msg = MIMEMultipart()
        msg["From"] = self.from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.from_email, self.from_password)
            server.sendmail(self.from_email, to_email, msg.as_string())
            server.quit()
            logger.info(f"Email sent successfully to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Email delivery failed to {to_email}: {e}")
            raise EmailDeliveryError(to_email, str(e))

    def send_username_recovery(self, to_email: str, username: str) -> bool:
        """Send a username recovery email."""
        subject = "PharmaStock — Username Recovery"
        body = (
            f"Dear user,\n\n"
            f"Your username is: {username}\n\n"
            f"Best regards,\n"
            f"PharmaStock Optimizer Team"
        )
        return self.send(to_email, subject, body)
