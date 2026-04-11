"""
Email dataset for the Email Triage OpenEnv environment.
Each email has ground-truth labels used by graders.
"""

from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Priority adjacency map for partial credit (easy task)
# ---------------------------------------------------------------------------
PRIORITY_ORDER = ["spam", "low", "normal", "high", "urgent"]

def priority_partial_score(predicted: str, gold: str) -> float:
    """Give partial credit based on distance between priority levels."""
    if predicted == gold:
        return 1.0
    try:
        pi = PRIORITY_ORDER.index(predicted)
        gi = PRIORITY_ORDER.index(gold)
        dist = abs(pi - gi)
        if dist == 1:
            return 0.5
        elif dist == 2:
            return 0.2
        else:
            return 0.0
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# TASK EASY — 10 emails, priority classification only
# ---------------------------------------------------------------------------
EASY_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "e001",
        "subject": "URGENT: Production database down — customers cannot checkout",
        "sender": "alice@ops.acme.com",
        "sender_domain": "ops.acme.com",
        "body": (
            "Hi team,\n\nOur production database went down 10 minutes ago. "
            "All checkout flows are failing with 500 errors. Revenue impact is ~$5k/min. "
            "Need all hands on deck immediately. Please join the war room call NOW.\n\nAlice"
        ),
        "timestamp": "2024-03-15T09:02:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "urgent",
            "category": "it",
            "action": "escalate",
        },
    },
    {
        "email_id": "e002",
        "subject": "Congratulations! You've won a $500 Amazon gift card",
        "sender": "noreply@prize-winner-2024.xyz",
        "sender_domain": "prize-winner-2024.xyz",
        "body": (
            "Dear valued customer,\n\nYou have been selected as this month's lucky winner! "
            "Click here to claim your $500 Amazon gift card: http://bit.ly/cl41m-pr1ze\n\n"
            "Hurry, offer expires in 24 hours!"
        ),
        "timestamp": "2024-03-15T09:05:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "spam",
            "category": "spam",
            "action": "delete",
        },
    },
    {
        "email_id": "e003",
        "subject": "Q1 All-Hands meeting — save the date",
        "sender": "hr@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "Hi everyone,\n\nPlease save the date for our Q1 All-Hands on April 5th at 2pm PT. "
            "Calendar invite to follow. Light refreshments will be served in the office.\n\nHR Team"
        ),
        "timestamp": "2024-03-15T09:10:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "low",
            "category": "internal",
            "action": "archive",
        },
    },
    {
        "email_id": "e004",
        "subject": "Re: Invoice #4821 — payment not received",
        "sender": "billing@vendor-corp.com",
        "sender_domain": "vendor-corp.com",
        "body": (
            "Hello,\n\nThis is a follow-up regarding invoice #4821 for $12,450 due on March 1st. "
            "We have not received payment and this is now 14 days overdue. "
            "Please arrange payment or contact us to discuss. "
            "Failure to respond within 5 business days may result in service suspension.\n\nRegards,\nBilling Dept"
        ),
        "timestamp": "2024-03-15T09:15:00Z",
        "thread_length": 3,
        "has_attachments": True,
        "gold": {
            "priority": "high",
            "category": "billing",
            "action": "forward",
        },
    },
    {
        "email_id": "e005",
        "subject": "Quick question about your API rate limits",
        "sender": "dev@startup-xyz.io",
        "sender_domain": "startup-xyz.io",
        "body": (
            "Hi,\n\nI'm integrating your API and noticed the docs say 1000 req/min but "
            "I'm hitting limits at around 800. Is this expected? Any chance of a higher tier?\n\nThanks,\nJordan"
        ),
        "timestamp": "2024-03-15T09:20:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "normal",
            "category": "support",
            "action": "reply",
        },
    },
    {
        "email_id": "e006",
        "subject": "Security alert: new sign-in from unknown device",
        "sender": "security@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "We detected a sign-in to your account from a new device:\n"
            "Location: Lagos, Nigeria\nDevice: Windows PC\nTime: 2024-03-15 08:55 UTC\n\n"
            "If this was not you, please reset your password immediately and contact IT Security."
        ),
        "timestamp": "2024-03-15T09:00:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "urgent",
            "category": "it",
            "action": "escalate",
        },
    },
    {
        "email_id": "e007",
        "subject": "Monthly newsletter — March 2024",
        "sender": "newsletter@industry-digest.com",
        "sender_domain": "industry-digest.com",
        "body": (
            "Welcome to the March 2024 Industry Digest!\n\n"
            "Top stories: AI regulation update, SaaS market trends, funding roundup...\n"
            "[Read more online]"
        ),
        "timestamp": "2024-03-15T08:00:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "low",
            "category": "other",
            "action": "archive",
        },
    },
    {
        "email_id": "e008",
        "subject": "Complaint: order #78234 arrived damaged",
        "sender": "customer.jane@gmail.com",
        "sender_domain": "gmail.com",
        "body": (
            "Hello,\n\nI received my order #78234 yesterday and the item was completely smashed. "
            "The packaging was intact so this happened before shipping. "
            "I need a replacement ASAP — I ordered this as a birthday gift. "
            "Very disappointed with the quality control.\n\nJane Miller"
        ),
        "timestamp": "2024-03-15T09:30:00Z",
        "thread_length": 1,
        "has_attachments": True,
        "gold": {
            "priority": "high",
            "category": "complaint",
            "action": "reply",
        },
    },
    {
        "email_id": "e009",
        "subject": "Can we schedule a demo next week?",
        "sender": "sarah.jones@prospect-co.com",
        "sender_domain": "prospect-co.com",
        "body": (
            "Hi,\n\nI saw your product at TechConf last week and I'm interested in learning more. "
            "Could we schedule a 30-minute demo sometime next week? "
            "We have a team of 200 and are looking to replace our current solution.\n\nBest,\nSarah"
        ),
        "timestamp": "2024-03-15T09:45:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "high",
            "category": "sales",
            "action": "reply",
        },
    },
    {
        "email_id": "e010",
        "subject": "Please review updated vacation policy doc",
        "sender": "hr@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "Hi team,\n\nAttached is the updated vacation policy effective April 1st. "
            "Key change: PTO rollover cap increased from 5 to 10 days. "
            "Please review and acknowledge by replying to this email.\n\nHR"
        ),
        "timestamp": "2024-03-15T10:00:00Z",
        "thread_length": 1,
        "has_attachments": True,
        "gold": {
            "priority": "normal",
            "category": "hr",
            "action": "reply",
        },
    },
]

# ---------------------------------------------------------------------------
# TASK MEDIUM — 10 emails, priority + category + action
# ---------------------------------------------------------------------------
MEDIUM_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "m001",
        "subject": "Legal notice: cease and desist",
        "sender": "counsel@lawfirm-partners.com",
        "sender_domain": "lawfirm-partners.com",
        "body": (
            "Dear Sir/Madam,\n\nWe represent XYZ Corp and write to demand you immediately cease "
            "use of the trademark 'SwiftDash' which infringes our client's registered mark. "
            "Failure to comply within 10 days will result in legal action.\n\nAttorney at Law"
        ),
        "timestamp": "2024-03-15T10:05:00Z",
        "thread_length": 1,
        "has_attachments": True,
        "gold": {
            "priority": "urgent",
            "category": "legal",
            "action": "escalate",
        },
    },
    {
        "email_id": "m002",
        "subject": "Re: Onboarding — laptop not set up",
        "sender": "new.hire.bob@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "Hi,\n\nI'm starting tomorrow and just realised I never received instructions for "
            "setting up my laptop. I don't have access to Slack or email properly yet. "
            "Could someone help? I don't want to start on the wrong foot.\n\nBob Chen, New Hire"
        ),
        "timestamp": "2024-03-15T10:10:00Z",
        "thread_length": 2,
        "has_attachments": False,
        "gold": {
            "priority": "high",
            "category": "it",
            "action": "forward",
        },
    },
    {
        "email_id": "m003",
        "subject": "Your subscription will renew in 3 days",
        "sender": "billing@saas-tool.com",
        "sender_domain": "saas-tool.com",
        "body": (
            "Hi,\n\nYour Pro plan ($299/month) will auto-renew on March 18th. "
            "No action needed if you wish to continue. To cancel or change your plan, "
            "visit your account settings.\n\nSaaS Tool Team"
        ),
        "timestamp": "2024-03-15T10:15:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "normal",
            "category": "billing",
            "action": "archive",
        },
    },
    {
        "email_id": "m004",
        "subject": "Interview scheduling — Senior Engineer role",
        "sender": "recruiting@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "Hi,\n\nWe'd like to schedule your technical interview for the Senior Engineer "
            "position. Please use this link to pick a 90-minute slot: calendly.com/acme/tech-interview\n\n"
            "Interview panel: 3 engineers + hiring manager.\n\nRecruiting Team"
        ),
        "timestamp": "2024-03-15T10:20:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "high",
            "category": "hr",
            "action": "reply",
        },
    },
    {
        "email_id": "m005",
        "subject": "Customer data export request — GDPR Article 20",
        "sender": "privacy@eu-regulator.eu",
        "sender_domain": "eu-regulator.eu",
        "body": (
            "Dear Data Controller,\n\nWe have received a data portability request under GDPR Art. 20 "
            "from a data subject. You are required to provide all personal data held within 30 days. "
            "Ref: GDPR-2024-04821.\n\nRegards,\nData Protection Authority"
        ),
        "timestamp": "2024-03-15T10:25:00Z",
        "thread_length": 1,
        "has_attachments": True,
        "gold": {
            "priority": "urgent",
            "category": "legal",
            "action": "escalate",
        },
    },
    {
        "email_id": "m006",
        "subject": "Thanks for the great support yesterday!",
        "sender": "happy.customer@example.com",
        "sender_domain": "example.com",
        "body": (
            "Hi,\n\nJust wanted to say your support agent Maria was fantastic yesterday. "
            "She resolved my billing issue in under 10 minutes. Please pass on my thanks!\n\nMike"
        ),
        "timestamp": "2024-03-15T10:30:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "low",
            "category": "support",
            "action": "archive",
        },
    },
    {
        "email_id": "m007",
        "subject": "Partnership proposal — co-marketing opportunity",
        "sender": "partnerships@bigco.com",
        "sender_domain": "bigco.com",
        "body": (
            "Hi,\n\nWe're exploring co-marketing opportunities with complementary SaaS tools. "
            "BigCo has 50k customers in your target segment. Would you be open to a call "
            "this week to explore a joint webinar + newsletter swap?\n\nPartnerships Team, BigCo"
        ),
        "timestamp": "2024-03-15T10:35:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "normal",
            "category": "sales",
            "action": "forward",
        },
    },
    {
        "email_id": "m008",
        "subject": "Expense report approval needed — $4,200 conference travel",
        "sender": "finance@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "Hi,\n\nExpense report ER-2024-0312 from Tom W. is awaiting your approval. "
            "Total: $4,200 (flights, hotel, meals) for SaaStr conference. "
            "Approval deadline: EOD today to process in this pay period.\n\nFinance"
        ),
        "timestamp": "2024-03-15T10:40:00Z",
        "thread_length": 1,
        "has_attachments": True,
        "gold": {
            "priority": "high",
            "category": "internal",
            "action": "reply",
        },
    },
    {
        "email_id": "m009",
        "subject": "Unsubscribe me from your mailing list",
        "sender": "angry.user@hotmail.com",
        "sender_domain": "hotmail.com",
        "body": (
            "I have clicked unsubscribe THREE times and I'm still getting your emails. "
            "This is illegal under CAN-SPAM. Remove me NOW or I'm reporting you to the FTC.\n\n"
        ),
        "timestamp": "2024-03-15T10:45:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "high",
            "category": "complaint",
            "action": "escalate",
        },
    },
    {
        "email_id": "m010",
        "subject": "Weekly engineering standup notes",
        "sender": "eng-bot@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "Weekly standup summary — week of March 11:\n"
            "- Auth service migration: 80% complete\n"
            "- Mobile app v2.3 shipped\n"
            "- 3 critical bugs closed\n"
            "- Next week: load testing, documentation sprint\n"
        ),
        "timestamp": "2024-03-15T08:00:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "low",
            "category": "internal",
            "action": "archive",
        },
    },
]

# ---------------------------------------------------------------------------
# TASK HARD — 15 emails, full triage + reply drafting required for some
# ---------------------------------------------------------------------------
HARD_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "h001",
        "subject": "CRITICAL: customer PII data leak reported",
        "sender": "security@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "URGENT — A researcher has reported that customer PII (names, emails, hashed passwords) "
            "is accessible via a misconfigured S3 bucket. Estimated exposure: 14,000 records. "
            "We need immediate action: take the bucket private, notify legal, begin breach assessment. "
            "Do NOT communicate externally yet.\n\nSecurity Team"
        ),
        "timestamp": "2024-03-15T11:00:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "urgent",
            "category": "legal",
            "action": "escalate",
            "reply_required": False,
            "reply_keywords": [],
        },
    },
    {
        "email_id": "h002",
        "subject": "Re: Refund request #9921 — still unresolved after 3 weeks",
        "sender": "frustrated.customer@gmail.com",
        "sender_domain": "gmail.com",
        "body": (
            "This is my fourth email. I requested a refund for order #9921 on Feb 22nd and "
            "I've received nothing but auto-replies. The amount is $349. I'm disputing with my "
            "credit card company if this isn't resolved today. This is completely unacceptable.\n\n"
            "- David Park"
        ),
        "timestamp": "2024-03-15T11:05:00Z",
        "thread_length": 4,
        "has_attachments": False,
        "gold": {
            "priority": "urgent",
            "category": "complaint",
            "action": "reply",
            "reply_required": True,
            "reply_keywords": ["apologize", "refund", "resolve", "escalate", "48 hours"],
        },
    },
    {
        "email_id": "h003",
        "subject": "Inquiry: do you support SSO with Okta?",
        "sender": "it.procurement@enterprise-co.com",
        "sender_domain": "enterprise-co.com",
        "body": (
            "Hi,\n\nWe're evaluating your platform for 800 seats. Before we proceed, "
            "we need to confirm: do you support SSO via Okta? Also SCIM provisioning? "
            "We can't proceed without both.\n\nIT Procurement"
        ),
        "timestamp": "2024-03-15T11:10:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "high",
            "category": "sales",
            "action": "reply",
            "reply_required": True,
            "reply_keywords": ["SSO", "Okta", "SCIM", "enterprise", "demo"],
        },
    },
    {
        "email_id": "h004",
        "subject": "Your account has been compromised — verify now",
        "sender": "security-alert@acme-secure.net",
        "sender_domain": "acme-secure.net",
        "body": (
            "Dear user,\n\nWe detected suspicious activity on your account. "
            "To avoid suspension, verify your identity immediately: "
            "http://acme-secure.net/verify?token=abc123\n\n"
            "If you do not verify within 2 hours your account will be locked.\n\nSecurity Team"
        ),
        "timestamp": "2024-03-15T11:15:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "spam",
            "category": "spam",
            "action": "delete",
            "reply_required": False,
            "reply_keywords": [],
            "is_phishing": True,
        },
    },
    {
        "email_id": "h005",
        "subject": "Performance improvement plan — confidential",
        "sender": "hr-confidential@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "Hi,\n\nThis email contains confidential HR information regarding a PIP for "
            "employee ID #4521. Please review the attached document and schedule a meeting "
            "with the employee's manager by March 20th. Do not share outside HR and management.\n\nHR"
        ),
        "timestamp": "2024-03-15T11:20:00Z",
        "thread_length": 1,
        "has_attachments": True,
        "gold": {
            "priority": "high",
            "category": "hr",
            "action": "flag",
            "reply_required": False,
            "reply_keywords": [],
        },
    },
    {
        "email_id": "h006",
        "subject": "Invoice attached — NET30",
        "sender": "invoices@design-studio.co",
        "sender_domain": "design-studio.co",
        "body": (
            "Hi,\n\nPlease find attached invoice INV-2024-088 for $3,200 for the brand refresh "
            "project completed in February. Payment due within 30 days.\n\nThank you!\nDesign Studio"
        ),
        "timestamp": "2024-03-15T11:25:00Z",
        "thread_length": 1,
        "has_attachments": True,
        "gold": {
            "priority": "normal",
            "category": "billing",
            "action": "forward",
            "reply_required": False,
            "reply_keywords": [],
        },
    },
    {
        "email_id": "h007",
        "subject": "Can you cover my on-call shift Saturday?",
        "sender": "colleague.tom@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "Hey,\n\nI have a family emergency and can't cover on-call this Saturday (March 16). "
            "Any chance you can swap? I'll take your April 6th shift in return.\n\nTom"
        ),
        "timestamp": "2024-03-15T11:30:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "normal",
            "category": "internal",
            "action": "reply",
            "reply_required": True,
            "reply_keywords": ["Saturday", "shift", "cover", "swap"],
        },
    },
    {
        "email_id": "h008",
        "subject": "Media inquiry — comment on recent outage",
        "sender": "reporter.jane@technews.com",
        "sender_domain": "technews.com",
        "body": (
            "Hi,\n\nI'm a reporter at TechNews covering your service outage on March 14th "
            "which affected thousands of users. I'd like an official comment for my article "
            "publishing tomorrow. Deadline is 5pm today.\n\nJane Doe, TechNews"
        ),
        "timestamp": "2024-03-15T11:35:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "urgent",
            "category": "legal",
            "action": "escalate",
            "reply_required": False,
            "reply_keywords": [],
        },
    },
    {
        "email_id": "h009",
        "subject": "Re: Bug report — data corruption in export",
        "sender": "power.user@bigclient.com",
        "sender_domain": "bigclient.com",
        "body": (
            "Following up on my report from last week (ticket #TK-8821). "
            "The CSV export is still corrupting decimal values — columns shift when the "
            "value contains a comma. This is blocking our finance team's month-end close. "
            "We need a hotfix or workaround urgently.\n\nPower User, Big Client"
        ),
        "timestamp": "2024-03-15T11:40:00Z",
        "thread_length": 3,
        "has_attachments": True,
        "gold": {
            "priority": "urgent",
            "category": "support",
            "action": "reply",
            "reply_required": True,
            "reply_keywords": ["ticket", "workaround", "engineer", "hotfix", "priority"],
        },
    },
    {
        "email_id": "h010",
        "subject": "Congrats on the Series B!",
        "sender": "investor.friend@vc-firm.com",
        "sender_domain": "vc-firm.com",
        "body": (
            "Just saw the TechCrunch announcement — congrats on closing the Series B! "
            "Well deserved. Let's grab coffee next time you're in SF.\n\nCheers,\nMike"
        ),
        "timestamp": "2024-03-15T11:45:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "low",
            "category": "other",
            "action": "reply",
            "reply_required": True,
            "reply_keywords": ["thank", "coffee", "catch up"],
        },
    },
    {
        "email_id": "h011",
        "subject": "Access request: production database read replica",
        "sender": "analyst.priya@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "Hi,\n\nI'm working on the Q1 analytics report and need read-only access to "
            "the production DB replica. My manager (Carlos) has approved this. "
            "Could someone from IT grant access? I need it by Wednesday.\n\nPriya"
        ),
        "timestamp": "2024-03-15T11:50:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "normal",
            "category": "it",
            "action": "forward",
            "reply_required": False,
            "reply_keywords": [],
        },
    },
    {
        "email_id": "h012",
        "subject": "Automatic reply: Jane Smith is out of office",
        "sender": "jane.smith@partner.com",
        "sender_domain": "partner.com",
        "body": (
            "Thanks for your email. I'm out of office March 15–22 with limited access. "
            "For urgent matters, contact my colleague at backup@partner.com.\n\nJane"
        ),
        "timestamp": "2024-03-15T11:55:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "low",
            "category": "other",
            "action": "archive",
            "reply_required": False,
            "reply_keywords": [],
        },
    },
    {
        "email_id": "h013",
        "subject": "Harassment complaint — formal report",
        "sender": "employee.anonymous@acme.com",
        "sender_domain": "acme.com",
        "body": (
            "I'm submitting a formal harassment complaint against my manager. "
            "I have documented 4 incidents over the past 6 weeks. "
            "I request this be handled confidentially and escalated to HR leadership immediately. "
            "I'm also considering contacting an employment lawyer if not addressed this week."
        ),
        "timestamp": "2024-03-15T12:00:00Z",
        "thread_length": 1,
        "has_attachments": True,
        "gold": {
            "priority": "urgent",
            "category": "hr",
            "action": "escalate",
            "reply_required": False,
            "reply_keywords": [],
        },
    },
    {
        "email_id": "h014",
        "subject": "Feature request: dark mode for mobile app",
        "sender": "feature.fan@gmail.com",
        "sender_domain": "gmail.com",
        "body": (
            "Hi,\n\nLove your product! One thing I'd really like is dark mode on the mobile app. "
            "I use it at night and the brightness is brutal. Any plans for this?\n\nThanks!"
        ),
        "timestamp": "2024-03-15T12:05:00Z",
        "thread_length": 1,
        "has_attachments": False,
        "gold": {
            "priority": "low",
            "category": "support",
            "action": "reply",
            "reply_required": True,
            "reply_keywords": ["feature", "roadmap", "feedback", "noted"],
        },
    },
    {
        "email_id": "h015",
        "subject": "Re: Contract renewal — terms under review",
        "sender": "legal@enterprise-client.com",
        "sender_domain": "enterprise-client.com",
        "body": (
            "Dear Account Team,\n\nWe are in review of the contract renewal terms you sent. "
            "Our legal team has flagged clauses 4.2 (liability cap) and 7.1 (data residency). "
            "We cannot sign until these are revised. Current contract expires April 1st — "
            "we need a revised draft by March 22nd.\n\nLegal Dept, Enterprise Client"
        ),
        "timestamp": "2024-03-15T12:10:00Z",
        "thread_length": 5,
        "has_attachments": True,
        "gold": {
            "priority": "urgent",
            "category": "legal",
            "action": "escalate",
            "reply_required": False,
            "reply_keywords": [],
        },
    },
]

TASK_EMAILS = {
    "task_easy": EASY_EMAILS,
    "task_medium": MEDIUM_EMAILS,
    "task_hard": HARD_EMAILS,
}