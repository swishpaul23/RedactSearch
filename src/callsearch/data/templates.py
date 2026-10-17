"""Finance call templates and MultiWOZ injection phrases with {TOKEN} slots, split train/test.

Slots: PII -> PERSON (customer full name), FIRST (customer first name), AGENT (agent first
name), CARD, SIN, PHONE, EMAIL, ACCOUNT. Distractors -> ORDER (16-digit, fails Luhn),
REF (9-digit, fails Luhn), AMOUNT, DATE, TIME. A trailing digit (AMOUNT2) is a second,
independent value of the same kind. Values are cached per call.

Test templates and phrases are held out: different wording from train, never shared.
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
        (A, "Perfect, you're all set, {FIRST}."),
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
        (C, "{PERSON}."),
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
]

# --- MultiWOZ reshaping phrases ----------------------------------------------
# GREETING is prepended as a new agent turn; the rest are appended to existing turns.
INJECTIONS = {
    "train": {
        "GREETING": [
            "Thanks for calling the Cambridge travel line, this is {AGENT}.",
            "Hello, you've reached the booking desk, my name is {AGENT}.",
        ],
        "PERSON": [
            "It's under the name {PERSON}.",
            "My name is {PERSON}, by the way.",
            "You can put it under {PERSON}.",
        ],
        "PHONE": [
            "You can reach me at {PHONE} if anything changes.",
            "My callback number is {PHONE}.",
        ],
        "EMAIL": [
            "Please send the confirmation to {EMAIL}.",
            "My email is {EMAIL} if you need it.",
        ],
        "CARD": [
            "You can hold it with my card, {CARD}.",
            "Here's a card to secure the booking: {CARD}.",
        ],
        "DISTRACTOR_CUSTOMER": [
            "I still have the order number from my last booking, {ORDER}, if that helps.",
            "Last time the deposit was {AMOUNT}, is that still right?",
        ],
        "DISTRACTOR_AGENT": [
            "Your booking reference is {REF}.",
            "That comes to {AMOUNT}, payable on {DATE}.",
        ],
    },
    "test": {
        "GREETING": [
            "Good day, reservations, {AGENT} speaking.",
        ],
        "PERSON": [
            "The booking should be for {PERSON}.",
            "Put me down as {PERSON}.",
        ],
        "PHONE": [
            "If you need me, my number is {PHONE}.",
            "Text me at {PHONE} please.",
        ],
        "EMAIL": [
            "Could you email the details to {EMAIL}?",
        ],
        "CARD": [
            "Go ahead and charge the deposit to {CARD}.",
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
