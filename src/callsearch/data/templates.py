"""Finance call templates and MultiWOZ injection phrases with {TOKEN} slots, split train/dev/test.

Slots: PII -> PERSON (customer full name), FIRST, LAST (customer first / last name), AGENT
(agent first name), CARD, SIN, PHONE, EMAIL, ACCOUNT. TITLE (Mr./Ms./Dr. ...) is plain,
unlabeled text. Distractors -> ORDER (16-digit, fails Luhn), REF (9-digit, fails Luhn),
AMOUNT, DATE, TIME. A trailing digit (PHONE2) is a second, independent value of the same
kind. Values are cached per call.

Every split has its own templates and phrase pools; no line is shared between splits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    id: str
    topic: str
    split: str
    lines: tuple[tuple[str, str], ...]


A, C = "agent", "customer"

TRAIN_TEMPLATES = [
    Template("refund_a", "refund", "train", (
        (A, "Thanks for calling Northshore Financial, my name is {AGENT}. Who am I speaking with?"),
        (C, "Hi, this is {PERSON}. I'm calling about a refund for an order I cancelled."),
        (A, "Sorry to hear that, {FIRST}. Do you have the order number?"),
        (C, "Yes, it's {ORDER}. I cancelled it on {DATE} but I was still charged {AMOUNT}."),
        (A, "I see it. Which card was the charge on?"),
        (C, "It went on my card, {CARD}."),
        (A, "Thank you. I've submitted the refund of {AMOUNT} back to that card. It can take five to seven business days."),
        (C, "Great. Can you email me a confirmation at {EMAIL}?"),
        (A, "Done. Your refund reference is {REF}. Anything else today?"),
        (C, "No, that's everything. Thanks {AGENT}."),
    )),
    Template("billing_a", "billing_dispute", "train", (
        (A, "Good afternoon, billing department, this is {AGENT} speaking."),
        (C, "Hi {AGENT}, I want to dispute a charge on my statement."),
        (A, "Of course. Can I get your full name and account number?"),
        (C, "{PERSON}, and my account number is {ACCOUNT}."),
        (A, "Thanks. Which charge are you disputing?"),
        (C, "There's a charge of {AMOUNT} from {DATE} that I don't recognize. My usual bill is around {AMOUNT2}."),
        (A, "I can open a dispute for that. What's the best number to reach you?"),
        (C, "You can call me at {PHONE}."),
        (A, "So that's {PHONE}. Your dispute reference number is {REF} and we'll follow up by {DATE2}."),
        (C, "Perfect, thank you."),
    )),
    Template("card_replace_a", "card_replacement", "train", (
        (A, "Thank you for calling card services, I'm {AGENT}. How can I help?"),
        (C, "Hi, I lost my credit card yesterday and I need a replacement."),
        (A, "I'm sorry about that. Can you confirm your name for me?"),
        (C, "Sure, it's {PERSON}."),
        (A, "Thanks {FIRST}. Do you have the card number, or should I look it up?"),
        (C, "I wrote it down, it's {CARD}."),
        (A, "Got it, I've blocked that card. Were there any purchases you didn't make?"),
        (C, "I don't think so. The last thing I bought was about {AMOUNT} on {DATE}."),
        (A, "That matches. The new card ships today. Should we text updates to {PHONE}?"),
        (C, "Yes, that's still my number."),
        (A, "Great. Your case number is {REF}. Is there anything else?"),
        (C, "No, thanks for the help."),
    )),
    Template("identity_a", "identity_verification", "train", (
        (A, "Northshore lending, this is {AGENT}. Are you calling about a loan application?"),
        (C, "Yes, my name is {PERSON} and I applied for a personal loan on {DATE}."),
        (A, "I can help with that. Before I continue I need to verify your identity. What is your social insurance number?"),
        (C, "It's {SIN}."),
        (A, "Thank you. And the account number you'd like the funds deposited to?"),
        (C, "The account number is {ACCOUNT}."),
        (A, "And the email on file?"),
        (C, "{EMAIL}."),
        (A, "Everything checks out, {FIRST}. The approved amount is {AMOUNT}. Your application number is {REF}."),
        (C, "Great. When will the money arrive?"),
        (A, "Usually within two business days, by {TIME} on the deposit day."),
        (C, "Thanks so much."),
    )),
    Template("refund_b", "refund", "train", (
        (A, "Hello, you've reached customer support, my name is {AGENT}."),
        (C, "Hi, I was charged twice for the same purchase and I'd like one of them refunded."),
        (A, "I can look into that. What's the name on the account?"),
        (C, "{PERSON}."),
        (A, "And the card the charges were on?"),
        (C, "My card number is {CARD}. Both charges were {AMOUNT} at about {TIME}."),
        (A, "I see both. I've reversed the duplicate. Your confirmation number is {REF}."),
        (C, "Thanks. Will I get something in writing?"),
        (A, "Yes, we'll send it to the email on file. Can you confirm it?"),
        (C, "It's {EMAIL}."),
        (A, "Perfect, you're all set, {TITLE} {LAST}."),
    )),
    Template("card_replace_b", "card_replacement", "train", (
        (A, "Card services, {AGENT} speaking. What can I do for you today?"),
        (C, "My debit card is cracked and the chip doesn't work anymore."),
        (A, "No problem, I can order a new one. Can I have your full name?"),
        (C, "{PERSON}."),
        (A, "For security, please confirm your social insurance number."),
        (C, "Sure, {SIN}."),
        (A, "Thank you. Which account is the card linked to?"),
        (C, "My chequing, account number is {ACCOUNT}."),
        (A, "Got it. There's a replacement fee of {AMOUNT}. The card should arrive by {DATE}."),
        (C, "That's fine. Can you send the tracking details to {EMAIL}?"),
        (A, "Absolutely. Your order number for the new card is {ORDER}."),
        (C, "Thank you."),
    )),
    # Customer speaks first; name given as title + surname, then in full.
    Template("loan_inquiry_a", "loan_inquiry", "train", (
        (C, "Hi, is this the loans line? I had a question about my line of credit."),
        (A, "It is, you're speaking with {AGENT}. Can I get the account number for the line of credit?"),
        (C, "Yeah, the account number is {ACCOUNT}."),
        (A, "And whose name is on it?"),
        (C, "Mine, {TITLE} {LAST}. Well, {PERSON} in full."),
        (A, "Thanks. The current balance is {AMOUNT} and the minimum payment of {AMOUNT2} is due {DATE}."),
        (C, "Can I pay the minimum now with my card? It's {CARD}."),
        (A, "I can process that now. Your payment confirmation is {REF}."),
        (C, "Perfect, that's all I needed."),
    )),
    # Long call: SIN verification, then two phones and an email.
    Template("contact_change_a", "contact_change", "train", (
        (A, "Member services, {AGENT} speaking, how can I help you today?"),
        (C, "I need to update my contact details. I got a new phone and a new email."),
        (A, "Sure. First I need to verify you. Full name please?"),
        (C, "It's {PERSON}, I've been a member for years."),
        (A, "And your social insurance number, please?"),
        (C, "Okay, it's {SIN}."),
        (A, "Thank you, {TITLE} {LAST}, you're verified. What's the new phone number?"),
        (C, "The new one is {PHONE}. The old one was {PHONE2}, you can delete that."),
        (A, "Done. And the new email?"),
        (C, "Put down {EMAIL}."),
        (A, "I've updated both as of {DATE}. You'll get a confirmation text shortly."),
        (C, "Great, thanks for your help today."),
    )),
    # Very short call.
    Template("account_inquiry_a", "account_inquiry", "train", (
        (A, "Northshore banking, {AGENT} here."),
        (C, "Hi, I just want to check whether my deposit cleared. Account {ACCOUNT}."),
        (A, "One moment. Can you confirm the name on the account?"),
        (C, "{PERSON}, it's a joint account but I'm the primary."),
        (A, "Yes, the deposit of {AMOUNT} cleared at {TIME}."),
        (C, "Oh good, that's a relief. Bye."),
    )),
    # Outbound call: the agent asks for the customer by title + surname.
    Template("billing_b", "billing_dispute", "train", (
        (A, "Hello, may I speak with {TITLE} {LAST}?"),
        (C, "Speaking."),
        (A, "Hi, this is {AGENT} from Northshore billing. I'm calling about a payment that didn't go through on {DATE}."),
        (C, "Oh no. Which payment?"),
        (A, "Your monthly payment of {AMOUNT}. Before I go on, can you confirm your full name?"),
        (C, "Sure, {PERSON}."),
        (A, "Thank you. Would you like to pay now with a card?"),
        (C, "Yes, use {CARD}."),
        (A, "That went through. Your receipt number is {REF}, and I'll email it to the address we have."),
        (C, "That's {EMAIL}, right?"),
        (A, "That's the one. Have a good day, {FIRST}."),
    )),
    # Short verification with surname only.
    Template("identity_b", "identity_verification", "train", (
        (A, "Security team, this is {AGENT}. We noticed a login from a new device."),
        (C, "That was me, I bought a new laptop."),
        (A, "No problem, I just need to verify it's you. What's your last name?"),
        (C, "{LAST}."),
        (A, "And your SIN?"),
        (C, "Uh, {SIN}."),
        (A, "All verified. Have a good one."),
    )),
    # Customer gives everything in one long turn; two agent turns in a row.
    Template("refund_c", "refund", "train", (
        (A, "Returns and refunds, {AGENT} speaking."),
        (C, "Hi, so my name's {PERSON}, my order number was {ORDER}, I paid {AMOUNT} on my card {CARD}, "
            "and the item never arrived, so I just want my money back."),
        (A, "Okay, that's a lot at once, thank you. Let me pull that up."),
        (A, "I see it. The courier marked it lost on {DATE}. I've issued the refund."),
        (C, "How long will it take?"),
        (A, "Three to five business days. I'll send the details to your email, which is?"),
        (C, "Oh, {EMAIL}."),
        (A, "Got it. Reference number {REF}."),
    )),
]

DEV_TEMPLATES = [
    # Customer first; surname with title, then full name.
    Template("refund_d", "refund", "dev", (
        (C, "Hello? Yes, hi. I'm trying to get a refund for a subscription I cancelled."),
        (A, "I can help. I'm {AGENT}. What name is the subscription under?"),
        (C, "{TITLE} {LAST}. That's {PERSON} in full."),
        (A, "Found you. You were billed {AMOUNT} on {DATE}, after the cancellation."),
        (C, "Exactly. That charge went to {CARD}."),
        (A, "Right, I'm reversing it now. Expect it within a week."),
        (C, "And you'll send something confirming that?"),
        (A, "Yes, to {EMAIL}. Confirmation code {REF}."),
        (C, "Wonderful, cheers."),
    )),
    Template("billing_d", "billing_dispute", "dev", (
        (A, "Billing inquiries, good evening. {AGENT} at your service."),
        (C, "Evening. There's a roaming charge on my bill and I never left the country."),
        (A, "Let's sort that out. The account it's on?"),
        (C, "The number on my account is {ACCOUNT}."),
        (A, "Thank you. And to confirm, am I speaking with the account holder?"),
        (C, "You are. {FIRST} here, well, {PERSON} officially."),
        (A, "I see the roaming charge of {AMOUNT} dated {DATE}. That looks like a system error."),
        (C, "So it's coming off?"),
        (A, "It is. If anything else comes up, we'll ring {PHONE}."),
        (C, "Sounds good."),
    )),
    # Identity check (SIN) happens before anything else.
    Template("card_replace_d", "card_replacement", "dev", (
        (A, "Lost and stolen cards, {AGENT}. Before we start, I'll need to verify your identity."),
        (C, "Okay, go ahead."),
        (A, "Could you give me your SIN?"),
        (C, "Yeah, {SIN}."),
        (A, "Thanks. And the name as it appears on the card?"),
        (C, "As printed, {PERSON}."),
        (A, "{TITLE} {LAST}, I've cancelled the old card. Was the card number {CARD}?"),
        (C, "Yes, that's the one I lost."),
        (A, "Your new card ships {DATE}. Tracking will go to {EMAIL}."),
    )),
    # Caller asks on someone else's behalf.
    Template("loan_d", "loan_inquiry", "dev", (
        (A, "Thanks for holding. Lending, {AGENT} speaking."),
        (C, "Hi. I'm checking the status of a mortgage pre-approval."),
        (A, "Sure thing. What's the applicant's name?"),
        (C, "It's under my spouse's name, {PERSON}. I'm calling for them."),
        (A, "I'll need the applicant's account number to look it up."),
        (C, "Account no {ACCOUNT}."),
        (A, "The pre-approval for {AMOUNT} was signed off on {DATE}. Reference {REF}."),
        (C, "That's great news."),
    )),
    # Short, customer-led; surname first, then first name.
    Template("contact_change_d", "contact_change", "dev", (
        (C, "Hi there, I moved to Calgary and need to change my phone number on file."),
        (A, "Happy to help, this is {AGENT}. Your name?"),
        (C, "{LAST}. First name {FIRST}."),
        (A, "Thank you. The new number?"),
        (C, "{PHONE}. And my email stays the same."),
        (A, "Updated. A text will go to {PHONE} to confirm."),
    )),
]

TEST_TEMPLATES = [
    Template("refund_h", "refund", "test", (
        (A, "Hi there, {AGENT} here from the returns team."),
        (C, "Hello. I returned a jacket a couple of weeks ago and the money never came back."),
        (A, "Let me check. Who's the purchase under?"),
        (C, "It should be under {PERSON}. The receipt shows order {ORDER}."),
        (A, "Found it, {FIRST}. The return was received on {DATE}. What did you pay with?"),
        (C, "My credit card, the number on it is {CARD}."),
        (A, "Okay. The credit of {AMOUNT} was delayed, I've pushed it through now."),
        (C, "Can someone ring me if it doesn't show up? My cell is {PHONE}."),
        (A, "Noted. Your ticket is {REF}."),
        (C, "Appreciate it."),
    )),
    Template("billing_h", "billing_dispute", "test", (
        (A, "You're through to accounts, I'm {AGENT}."),
        (C, "Hey, my bill went way up this month and nobody told me why."),
        (A, "I'm happy to go through it. Can I start with your name, please?"),
        (C, "{PERSON}. Account no. {ACCOUNT}."),
        (A, "Thanks {FIRST}. I see a late fee of {AMOUNT} and a service charge of {AMOUNT2}."),
        (C, "I paid on {DATE}, so the late fee shouldn't be there."),
        (A, "You're right, the payment posted at {TIME}. I'll remove it. Where should I send the corrected statement?"),
        (C, "Send it to {EMAIL}, please."),
        (A, "Will do. Reference {REF} for this adjustment."),
        (C, "Thanks a lot."),
    )),
    Template("card_replace_h", "card_replacement", "test", (
        (A, "Fraud prevention line, {AGENT} speaking."),
        (C, "Hi, I think someone stole my wallet. I need to cancel my credit card."),
        (A, "Let's do that right away. Who am I speaking to?"),
        (C, "Name's {PERSON}."),
        (A, "Can you read me the card number if you have it?"),
        (C, "I have a photo of it: {CARD}."),
        (A, "That card is frozen now. I do need to confirm your SIN before issuing a new one."),
        (C, "Okay, {SIN}."),
        (A, "Thanks. The replacement goes out on {DATE}. Any charges you don't recognize?"),
        (C, "Yes, one for {AMOUNT} earlier today around {TIME}."),
        (A, "I've flagged it. If we need you we'll call {PHONE}, is that right?"),
        (C, "Yep, that works."),
    )),
    Template("identity_h", "identity_verification", "test", (
        (A, "Mortgage services, good morning, you're speaking with {AGENT}."),
        (C, "Morning. I got a letter asking me to verify some details for my mortgage renewal."),
        (A, "Sure. What name is the mortgage under?"),
        (C, "{PERSON}, and the letter mentions it's account {ACCOUNT}."),
        (A, "Thank you. I'll need your social insurance number to verify."),
        (C, "My SIN is {SIN}."),
        (A, "That matches our records, {FIRST}. What's a good callback number?"),
        (C, "Best number is {PHONE}, or you can email {EMAIL}."),
        (A, "Great. Your renewal payment will be {AMOUNT} starting {DATE}. Reference {REF}."),
        (C, "Okay, thank you."),
    )),
    # Verification by account + name in one turn; two emails.
    Template("contact_change_h", "contact_change", "test", (
        (A, "Profile updates, {AGENT} on the line."),
        (C, "Good morning, I'd like to swap the email on my profile for a new one."),
        (A, "Absolutely. For verification, give me your account number and full name."),
        (C, "Sure. Account number: {ACCOUNT}. Name is {PERSON}."),
        (A, "Thanks, {TITLE} {LAST}. What's the new address?"),
        (C, "Change it to {EMAIL2}. The old one, {EMAIL}, I don't use anymore."),
        (A, "Done. We'll also keep {PHONE} as your backup contact?"),
        (C, "Please do."),
    )),
]

# --- MultiWOZ reshaping phrases ----------------------------------------------
# GREETING is prepended as a new agent turn; AGENT_NAME is prepended to the agent turn
# after the customer gives their name; the rest are appended to existing turns.
INJECTIONS = {
    "train": {
        "GREETING": [
            "Thanks for calling the Cambridge travel line, this is {AGENT}.",
            "Hello, you've reached the booking desk, my name is {AGENT}.",
            "Cambridge visitor services, {AGENT} speaking, how can I help?",
            "You're through to {AGENT} at the travel desk.",
        ],
        "PERSON": [
            "It's under the name {PERSON}.",
            "My name is {PERSON}, by the way.",
            "You can put it under {PERSON}.",
            "This is {FIRST} speaking, sorry, I should have said.",
            "The reservation is for {TITLE} {LAST}.",
            "Last name {LAST}, first name {FIRST}.",
            "{FIRST} here, I'm the one organizing this.",
            "Book it for {PERSON} and a friend.",
            "I'm calling on behalf of {TITLE} {LAST}.",
            "Could you note {PERSON} as the main guest?",
            "The surname is {LAST}.",
            "Everyone calls me {FIRST}.",
        ],
        "AGENT_NAME": [
            "Thank you, {FIRST}.",
            "Lovely, {TITLE} {LAST}.",
            "Noted, {PERSON}.",
            "Okay {FIRST}, one moment.",
        ],
        "PHONE": [
            "You can reach me at {PHONE} if anything changes.",
            "My callback number is {PHONE}.",
            "Call {PHONE} if there's a problem.",
            "In case you need to get hold of me, it's {PHONE}.",
        ],
        "EMAIL": [
            "Please send the confirmation to {EMAIL}.",
            "My email is {EMAIL} if you need it.",
            "Can the receipt go to {EMAIL}?",
        ],
        "CARD": [
            "You can hold it with my card, {CARD}.",
            "Here's a card to secure the booking: {CARD}.",
            "Use the card {CARD} for the deposit.",
        ],
        "DISTRACTOR_CUSTOMER": [
            "I still have the order number from my last booking, {ORDER}, if that helps.",
            "Last time the deposit was {AMOUNT}, is that still right?",
            "I spent {AMOUNT} there on {DATE}, so I know it's good.",
        ],
        "DISTRACTOR_AGENT": [
            "Your booking reference is {REF}.",
            "That comes to {AMOUNT}, payable on {DATE}.",
            "The booking is held until {TIME}.",
        ],
    },
    "dev": {
        "GREETING": [
            "Welcome to the Cambridge info line, I'm {AGENT}.",
            "Hi, {AGENT} on the booking team, what can I do for you?",
        ],
        "PERSON": [
            "{PERSON} is the name for the booking.",
            "Use my surname, {LAST}, for the booking.",
            "Oh, and it should say {TITLE} {LAST} on the confirmation.",
            "I go by {FIRST}.",
            "Make it out to {PERSON}, please.",
            "We're both going, but list {PERSON} first.",
        ],
        "AGENT_NAME": [
            "Perfect, {TITLE} {LAST}, let me check.",
            "Alright {FIRST}, bear with me.",
        ],
        "PHONE": [
            "Ring me on {PHONE} with any updates.",
            "{PHONE} is the best way to contact me.",
        ],
        "EMAIL": [
            "Forward everything to {EMAIL}, thanks.",
            "I'd like a copy at {EMAIL}.",
        ],
        "CARD": [
            "Guarantee it on {CARD}, please.",
            "The card I'd like to use is {CARD}.",
        ],
        "DISTRACTOR_CUSTOMER": [
            "The last time I booked, the order was {ORDER}.",
            "Is it still around {AMOUNT} a night?",
        ],
        "DISTRACTOR_AGENT": [
            "Reference for that is {REF}.",
            "It'll be {AMOUNT} in total.",
        ],
    },
    "test": {
        "GREETING": [
            "Good day, reservations, {AGENT} speaking.",
            "Reservations line, my name's {AGENT}, how may I help?",
        ],
        "PERSON": [
            "Put me down as {PERSON}.",
            "The booking should be for {PERSON}.",
            "Mark it under {LAST}, that's {TITLE} {LAST}.",
            "{FIRST} is my first name if you need it.",
            "Can you register it to {PERSON}?",
            "Hold it for a {TITLE} {LAST}.",
        ],
        "AGENT_NAME": [
            "Cheers, {FIRST}.",
            "Very good, {TITLE} {LAST}.",
        ],
        "PHONE": [
            "If you need me, my number is {PHONE}.",
            "Text me at {PHONE} please.",
        ],
        "EMAIL": [
            "Could you email the details to {EMAIL}?",
            "Drop me a line at {EMAIL} once it's confirmed.",
        ],
        "CARD": [
            "Go ahead and charge the deposit to {CARD}.",
            "Bill it to {CARD}.",
        ],
        "DISTRACTOR_CUSTOMER": [
            "My previous order was {ORDER}, not sure if you need that.",
            "I think I paid {AMOUNT} on {DATE} last time.",
        ],
        "DISTRACTOR_AGENT": [
            "I've noted it under confirmation {REF}.",
            "Check-in starts at {TIME}.",
        ],
    },
}
