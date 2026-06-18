"""
Does the following:
 - Reads all incoming messages. Extracts the body of that mail and plugs it into ChatGPT using OpenAI's API.
 - Asks ChatGPT to check whether or not this is a request by a customer. 
 - If it's a request, it first checks whether or not it is part of a reply thread from before. If it is, it tries to assign the same label that was assigned previously (if it exists). If the label doesn't previously exist, it extracts the name from the signature and uses that as the label.
 - If it's a request and either a name can't be found or it's a fresh mail that isn't a reply, it then assigns an employee to it by assigning a label to it. It does this in round-robin manner. The round-robin manner is as follows. 
 Suppose there are 3 employees: Employee_1, Employee_2, Employee_3.
 Request_1 -> Employee_1 | Request_2 -> Employee_2 | Request_3 -> Employee_3 | Request_4 -> Employee_1 | Request_5 -> Employee_2 | ...
"""

# Standard Imports
import os
import sys
import json
import pickle
import time
from datetime import datetime, timedelta
from dateutil import parser
import schedule
import pytz
import requests
# Gmail API utils
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import google.auth.transport.requests
from googleapiclient.errors import HttpError
# OpenAI import for checking completion of requests
from openai import OpenAI
# bs4 to parse through the body of mails
from bs4 import BeautifulSoup
# for encoding/decoding messages in base64
from base64 import urlsafe_b64decode, urlsafe_b64encode
# for dealing with attachement MIME types
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from mimetypes import guess_type as guess_mime_type
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sensitive_info/assigner.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

ASSIGNMENTS_CACHE = "sensitive_info/assignments_cache.json"


def with_retry(func, retries=3, delay=5):
    """
    Retries a callable on network/HTTP errors up to `retries` times.
    """
    for attempt in range(1, retries + 1):
        try:
            return func()
        except (HttpError, ConnectionError, OSError) as e:
            logger.warning(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                logger.error("All retry attempts exhausted.", exc_info=True)
                raise


def save_assignments(assignments: dict):
    """
    Saves the current assignments dict to disk so they survive a restart.
    """
    try:
        with open(ASSIGNMENTS_CACHE, "w") as f:
            json.dump(assignments, f)
        logger.info("Assignments saved to cache.")
    except Exception as e:
        logger.error(f"Failed to save assignments cache: {e}", exc_info=True)


def load_assignments() -> dict | None:
    """
    Loads assignments from disk if a cache exists.
    """
    if os.path.exists(ASSIGNMENTS_CACHE):
        try:
            with open(ASSIGNMENTS_CACHE) as f:
                data = json.load(f)
            logger.info("Assignments restored from cache.")
            return data
        except Exception as e:
            logger.error(f"Failed to load assignments cache: {e}", exc_info=True)
    return None

def build_service(creds):
    """
    Builds a service with a fresh requests session every time.
    Prevents WinError 10053 caused by Windows killing idle sockets.
    """
    session = requests.Session()
    session.headers.update({"Connection" : "close"})
    authed_session = google.auth.transport.requests.AuthorizedSession(creds)
    return build("gmail", "v1", credentials=creds, requestBuilder=None)

def gmail_authenticate():
    creds = None
    token_path = "sensitive_info/token.pickle"
    cred_path = "sensitive_info/credentials.json"
    if os.path.exists(token_path):
        with open(token_path, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "wb") as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds, cache_discovery=False)


class AutoAssigner:
    """
    Automatically assigns incoming insurance requests to employee labels through round-robin scheduling.
    Each employee has two sublabels: NEW REQUEST and ACTIONED.
    """
    def __init__(self, service, list_of_employee_labels: list[str]):
        self.service = service
        self.client = OpenAI()
        self.employee_names = list_of_employee_labels
        self.assignments: dict[str, list[dict]] = {
            name: [] for name in list_of_employee_labels
        }

        self.label_name_map = {}
        self.new_request_label_ids = {}
        self.actioned_label_ids = {}
        self.actioned_id_to_employee = {}
        self.actioned_label_id_set = set()
        self.new_request_label_id_set = set()
        self.all_employee_label_ids = set()

        self._refresh_label_maps()

        self.round_robin = self._assign_labels_round_robin(list_of_employee_labels)

    def _refresh_label_maps(self):
        """
        Re-fetches and rebuilds all label maps from Gmail.
        Called at startup and at the start of every identify_and_assign run
        to avoid stale data from a broken connection.
        """
        label_id_map = self._get_label_id_map()

        if not label_id_map:
            raise RuntimeError(
                "Label map came back empty from Gmail. "
                "This usually means a broken connection — the script will restart."
            )

        normalized = {k.lower(): v for k, v in label_id_map.items()}
        self.label_name_map = {v: k for k, v in label_id_map.items()}

        self.new_request_label_ids = {}
        self.actioned_label_ids = {}
        self.actioned_id_to_employee = {}

        for name in self.employee_names:
            nr_key = f"{name}/NEW REQUEST".lower()
            ac_key = f"{name}/ACTIONED".lower()

            if nr_key not in normalized or ac_key not in normalized:
                raise ValueError(
                    f"Could not find sublabels for '{name}'. "
                    f"Ensure 'NEW REQUEST' and 'ACTIONED' sublabels exist."
                )

            self.new_request_label_ids[name] = normalized[nr_key]
            self.actioned_label_ids[name] = normalized[ac_key]
            self.actioned_id_to_employee[normalized[ac_key]] = name

        self.actioned_label_id_set = set(self.actioned_label_ids.values())
        self.new_request_label_id_set = set(self.new_request_label_ids.values())
        self.all_employee_label_ids = self.new_request_label_id_set | self.actioned_label_id_set

        logger.info(f"Label maps refreshed successfully. {len(label_id_map)} total Gmail labels found.")

    def _extract_body(self, payload, mime_type="text/plain") -> str:
        """
        Extracts body from a message payload for the given mime type.
        """
        if payload.get("mimeType") == mime_type:
            data = payload["body"].get("data", "")
            data = data.replace("-", "+").replace("_", "/")
            return urlsafe_b64decode(data).decode("utf-8")

        for part in payload.get("parts", []):
            result = self._extract_body(part, mime_type)
            if result:
                return result

        return ""

    def _is_reply(self, headers: list) -> bool:
        """
        Checks if the message is a reply by looking for In-Reply-To or References headers.
        """
        for header in headers:
            if header["name"] in ("In-Reply-To", "References"):
                return True
        return False

    def _get_last_sent_signature(self, thread_id: str) -> str:
        """
        Fetches the thread and extracts the signature from the last sent message.
        """
        thread = with_retry(
            lambda: self.service.users().threads().get(userId='me', id=thread_id).execute()
        )
        messages = thread.get("messages", [])

        for message in reversed(messages):
            if "SENT" in message.get("labelIds", []):
                body = self._extract_body(message["payload"], mime_type="text/html")
                soup = BeautifulSoup(body, "html.parser")
                signature = soup.find(class_="gmail_signature")
                if signature:
                    return signature.get_text(separator="\n").strip()

        return ""

    def _extract_name_from_signature(self, signature: str) -> str:
        """
        Uses GPT to extract the person's name from an email signature.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=20,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Extract only the person's full name from the email signature below. "
                        f"Return just the name, nothing else.\n\n{signature}"
                    )
                }
            ]
        )
        return response.choices[0].message.content.strip()

    def _is_request(self, body: str, subject: str = "") -> bool:
        """
        Checks if the incoming mail is an addition or deletion request for medical insurance.
        """
        soup = BeautifulSoup(body, "html.parser")

        for quote in soup.find_all(class_=["gmail_quote", "gmail_quote_container"]):
            quote.decompose()
        for signature in soup.find_all(class_="gmail_signature"):
            signature.decompose()
        plain_text = soup.get_text(separator="\n").strip()

        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"From the text below identify whether or not it is a request for addition, deletion, refund, cancellation, ammendment, correction, active lists, COC, ECard, ICP of medical insurance."
                        f"If it's a request, return YES, otherwise return NO, nothing else.\n\n{subject}\n{plain_text}"
                    )
                }
            ]
        )

        return response.choices[0].message.content.strip().upper().startswith("YES")

    def _get_label_id_map(self) -> dict[str, str]:
        """
        Returns a mapping of label name -> label ID.
        """
        result = with_retry(
            lambda: self.service.users().labels().list(userId='me').execute()
        )
        labels = {label['name']: label['id'] for label in result.get('labels', [])}
        logger.info(f"Fetched {len(labels)} labels from Gmail.")
        return labels

    def _assign_labels_round_robin(self, employee_names: list[str]):
        """
        Generator that cycles through employee names in a round-robin manner.
        """
        index = 0
        while True:
            yield employee_names[index % len(employee_names)]
            index += 1

    def identify_and_assign(self):
        # Refresh label maps at the start of every run to avoid stale data
        self._refresh_label_maps()

        cutoff = datetime.now() - timedelta(hours=2)
        cutoff_ms = int(cutoff.timestamp() * 1000)

        date_str = cutoff.strftime("%Y/%m/%d")
        messages = []
        page_token = None

        while True:
            result = with_retry(
                lambda: self.service.users().messages().list(
                    userId='me',
                    q=f"after:{date_str} AND in:inbox AND -from:no-reply@cosmosinsurance.bitrix24.ae AND -from:te-notification@takafulemarat.com AND -from:no-reply@tefollowup.com AND -from:noreply.alsagr@alsagrins.ae",
                    pageToken=page_token
                ).execute()
            )

            messages.extend(result.get('messages', []))
            page_token = result.get('nextPageToken')

            if not page_token:
                break

        logger.info(f"Found {len(messages)} messages to process.")

        for message in messages:
            try:
                txt = with_retry(
                    lambda: self.service.users().messages().get(
                        userId='me',
                        id=message['id']
                    ).execute()
                )

                if int(txt.get("internalDate", 0)) < cutoff_ms:
                    continue

                payload = txt["payload"]
                headers = payload["headers"]

                subject = next(
                    (d["value"] for d in headers if d["name"] == "Subject"),
                    ""
                )

                # ----------------------------
                # 1. Check THIS EMAIL labels
                # ----------------------------
                existing_labels = set(txt.get("labelIds", []))

                if existing_labels & self.new_request_label_id_set:
                    logger.info(f"Skipping (already ASSIGNED): {subject}")
                    continue

                # ----------------------------
                # 2. Fetch FULL THREAD
                # ----------------------------
                thread = with_retry(
                    lambda: self.service.users().threads().get(
                        userId='me',
                        id=txt["threadId"]
                    ).execute()
                )

                thread_messages = thread.get("messages", [])

                thread_label_ids = set()
                for thread_msg in thread_messages:
                    thread_label_ids.update(thread_msg.get("labelIds", []))

                # ----------------------------
                # 3. Ignore CC EMAILS anywhere in thread
                # ----------------------------
                if any(
                    self.label_name_map.get(label_id, "").lower() == "cc emails"
                    for label_id in thread_label_ids
                ):
                    logger.info(f"Skipping thread (CC EMAILS): {subject}")
                    continue

                # ----------------------------
                # 4. Check ACTIONED in thread
                # ----------------------------
                thread_actioned_overlap = thread_label_ids & self.actioned_label_id_set

                body = self._extract_body(payload)
                is_request = self._is_request(body, subject)
                logger.info(f"{subject} | is_request={is_request}")

                if is_request:
                    label_id = None
                    employee_name = None
                    is_reply = self._is_reply(headers) or len(thread_messages) > 1
                    msg_type = "NEW"

                    # ----------------------------
                    # 5. Reply logic
                    # ----------------------------
                    if len(thread_messages) > 1:
                        actioned_employee = None
                        if thread_actioned_overlap:
                            actioned_label_id = next(iter(thread_actioned_overlap))
                            actioned_employee = self.actioned_id_to_employee.get(actioned_label_id)

                        signature_employee = None
                        signature = self._get_last_sent_signature(txt["threadId"])
                        if signature:
                            name = self._extract_name_from_signature(signature)
                            if name in self.new_request_label_ids:
                                signature_employee = name

                        # Signature takes priority if both found and they conflict
                        if signature_employee and actioned_employee:
                            if signature_employee != actioned_employee:
                                logger.info(
                                    f"Reply -- signature ({signature_employee}) overrides "
                                    f"ACTIONED ({actioned_employee})"
                                )
                                
                            else:
                                logger.info(f"Reply -- signature and ACTIONED agree: {signature_employee}")
                            employee_name = signature_employee
                            msg_type = "REPLY (SIGN)"

                        elif signature_employee:
                            logger.info(f"Reply -- assigning via signature to {signature_employee}/NEW REQUEST")
                            employee_name = signature_employee
                            msg_type = "REPLY (SIGN)"

                        elif actioned_employee:
                            logger.info(f"Reply -- assigning via ACTIONED to {actioned_employee}/NEW REQUEST")
                            employee_name = actioned_employee
                            msg_type = "REPLY (LABEL)"

                        else:
                            logger.warning(f"Reply detected but no match found from signature or ACTIONED, using round-robin | {subject}")

                        if employee_name:
                            label_id = self.new_request_label_ids[employee_name]

                    # ----------------------------
                    # 6. Round-robin fallback
                    # ----------------------------
                    if label_id is None:
                        employee_name = next(self.round_robin)
                        label_id = self.new_request_label_ids[employee_name]
                        logger.info(f"Assigning via round-robin to {employee_name}/NEW REQUEST")
                        msg_type = "NEW"

                    with_retry(
                        lambda: self.service.users().messages().modify(
                            userId='me',
                            id=message['id'],
                            body={"addLabelIds": [label_id]}
                        ).execute()
                    )

                    if employee_name:
                        self.assignments[employee_name].append({
                            "id": message["id"],
                            "subject": subject,
                            "thread_id": txt["threadId"],
                            "type": msg_type
                        })

                    logger.info(f"Assigned label '{self.label_name_map.get(label_id, label_id)}' to: {subject}")

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                continue

    def print_assignments(self):
        for employee, emails in self.assignments.items():
            logger.info(f"{employee} ({len(emails)} assigned):")
            for email in emails:
                tag = email.get("type", "UNKONWN")
                logger.info(f"  {tag} {email['subject']}")

    def send_assignment_notification(self, our_email: str):
        """
        Sends a daily summary email to ebservices with all assignments made in this run.
        """
        total = sum(len(emails) for emails in self.assignments.values())

        if total == 0:
            logger.info("No assignments made today, skipping notification email.")
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        rows = ""
        for employee, emails in self.assignments.items():
            if not emails:
                continue
            for email in emails:
                tag = email.get("type", "UNKNOWN")
                rows += (
                    f"<tr>"
                    f"<td style='padding:6px 12px;border:1px solid #ddd;'>{employee}</td>"
                    f"<td style='padding:6px 12px;border:1px solid #ddd;'>{email['subject']}</td>"
                    f"<td style='padding:6px 12px;border:1px solid #ddd;text-align:center;'>{tag}</td>"
                    f"</tr>"
                )

        body = f"""
        <html><body style="font-family:Arial,sans-serif;color:#333;">
            <h2 style="color:#2c5f8a;">Daily Assignment Summary</h2>
            <p>Date: <strong>{now}</strong> &nbsp;|&nbsp; Total assigned: <strong>{total}</strong></p>
            <table style="border-collapse:collapse;width:100%;margin-top:12px;">
                <thead>
                    <tr style="background:#2c5f8a;color:white;">
                        <th style="padding:8px 12px;text-align:left;">Employee</th>
                        <th style="padding:8px 12px;text-align:left;">Subject</th>
                        <th style="padding:8px 12px;text-align:center;">Type</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <p style="margin-top:16px;font-size:12px;color:#888;">
                This is an automated summary from the Auto Assigner.
            </p>
        </body></html>
        """

        msg = MIMEMultipart()
        msg["From"] = our_email
        msg["To"] = our_email
        msg["Subject"] = f"Assignment Summary | {now} | {total} assigned"
        msg.attach(MIMEText(body, "html"))

        raw = urlsafe_b64encode(msg.as_bytes()).decode()
        try:
            with_retry(
                lambda: self.service.users().messages().send(
                    userId="me",
                    body={"raw": raw}
                ).execute()
            )
            logger.info(f"Assignment notification email sent to {our_email}.")
        except Exception as e:
            logger.error(f"Failed to send assignment notification email: {e}", exc_info=True)


if __name__ == "__main__":
    SCOPES = ["https://mail.google.com/"]
    our_email = "ebservices@cosmosinsurance.com"
    LIST_OF_EMPLOYEE_LABELS = ["Prakash Pantha", "Sana Faisal", "Vidyalaxmi"]
    UAE_TZ = pytz.timezone("Asia/Dubai")

    try:
        service = gmail_authenticate()
    except Exception as e:
        logger.error(f"Failed to authenticate with Gmail: {e}", exc_info=True)
        raise SystemExit(1)

    auto_assigner = AutoAssigner(service, LIST_OF_EMPLOYEE_LABELS)

    # Restore any assignments saved from a previous run (e.g. after a crash restart)
    cached = load_assignments()
    if cached:
        # Only restore keys that match the current employee list
        for name in LIST_OF_EMPLOYEE_LABELS:
            if name in cached:
                auto_assigner.assignments[name] = cached[name]

    def run_assign():
        try:
            logger.info("--- Starting assignment run ---")
            auto_assigner.service = gmail_authenticate()
            auto_assigner.identify_and_assign()
            auto_assigner.print_assignments()
            save_assignments(auto_assigner.assignments)
            logger.info("--- Assignment run complete ---")
        except Exception as e:
            logger.error(f"Assignment run failed: {e}", exc_info=True)
            logger.warning("Saving assignments and restarting script to recover...")
            save_assignments(auto_assigner.assignments)
            os.execv(sys.executable, [sys.executable] + sys.argv)

    def send_daily_summary():
        logger.info("--- Sending daily assignment summary ---")
        auto_assigner.send_assignment_notification(our_email)
        # Clear assignments and cache after sending so tomorrow starts fresh
        for key in auto_assigner.assignments:
            auto_assigner.assignments[key] = []
        if os.path.exists(ASSIGNMENTS_CACHE):
            os.remove(ASSIGNMENTS_CACHE)
        logger.info("--- Assignments reset for next day ---")

    # If there's a need to run immediately on startup uncomment the below line, else it runs every 2 hours.
    # run_assign()
    schedule.every(2).hours.do(run_assign)

    # Daily summary at 08:00 UAE time, tracked by date to prevent double-sending
    last_summary_date = None

    logger.info("Scheduler started. Assigning every 2 hours, daily summary at 08:00 UAE time.")

    while True:
        schedule.run_pending()

        now_uae = datetime.now(UAE_TZ)
        if now_uae.hour == 8 and now_uae.minute == 0 and last_summary_date != now_uae.date():
            last_summary_date = now_uae.date()
            send_daily_summary()

        time.sleep(30)