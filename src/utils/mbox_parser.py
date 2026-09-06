import mailbox
import email
from email.header import decode_header
import os
import re
from typing import List, Dict, Any

def decode_mime_header(header_value: str) -> str:
    """Decode encoded MIME header strings (e.g. Subject or From fields)."""
    if not header_value:
        return "N/A"
    decoded_fragments = []
    try:
        for fragment, encoding in decode_header(header_value):
            if isinstance(fragment, bytes):
                decoded_fragments.append(fragment.decode(encoding or 'utf-8', errors='ignore'))
            else:
                decoded_fragments.append(str(fragment))
        return " ".join(decoded_fragments)
    except Exception:
        return str(header_value)

def extract_email_body(msg: email.message.Message) -> str:
    """Extract plain text body from email message object."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode('utf-8', errors='ignore')
            elif content_type == "text/html" and not body and "attachment" not in content_disposition:
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode('utf-8', errors='ignore')
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode('utf-8', errors='ignore')
        else:
            body = msg.get_payload() or ""
    return body

def parse_mbox_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parses an MBOX file and extracts email records.
    Supports standard mailbox format and fallback text-split parsing.
    """
    emails = []
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MBOX file not found at {filepath}")
        
    try:
        # Standard mailbox parse
        mbox = mailbox.mbox(filepath)
        for msg in mbox:
            subject = decode_mime_header(msg.get("Subject", "No Subject"))
            sender = decode_mime_header(msg.get("From", "Unknown Sender"))
            date = msg.get("Date", "Unknown Date")
            msg_id = msg.get("Message-ID", f"msg_{len(emails)+1}")
            body = extract_email_body(msg)
            
            if not body and not subject:
                continue
                
            emails.append({
                "Time": date,
                "Subject": subject,
                "Sender": sender,
                "Message-ID": msg_id,
                "Body": body
            })
            
    except Exception:
        # Fallback text parsing if mailbox format has custom delimiters
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        raw_emails = re.split(r'\nFrom\s+', content)
        for idx, raw_msg in enumerate(raw_emails):
            if not raw_msg.strip():
                continue
            lines = raw_msg.split('\n')
            subject = "No Subject"
            sender = "Unknown Sender"
            date = "Unknown Date"
            body_lines = []
            is_body = False
            
            for line in lines:
                if not is_body:
                    if line.lower().startswith("subject:"):
                        subject = line[8:].strip()
                    elif line.lower().startswith("from:"):
                        sender = line[5:].strip()
                    elif line.lower().startswith("date:"):
                        date = line[5:].strip()
                    elif line == "":
                        is_body = True
                else:
                    body_lines.append(line)
                    
            body = "\n".join(body_lines).strip()
            emails.append({
                "Time": date,
                "Subject": subject,
                "Sender": sender,
                "Message-ID": f"msg_{idx+1}",
                "Body": body if body else raw_msg
            })
            
    return emails
