"""Root authoring of the SLURP semantic knowledge sources for the NPU port (zero model contacts).

The historical authored sources (prepared/SAEA-SLURP-SEMANTIC-AUTHORING-PREPARE-V1/authored-intent-boundaries-v2.json,
authored-slot-contracts-v1.json and PREPARE-V2/authored-slot-contracts-v2.json) were never shipped. This script
re-authors them by the documented procedure: the root reads the governed TRAIN support rows of
prepared/SAEA-SLURP-SEMANTIC-AUTHORING-PREPARE-V1/support.json and writes the guidance values; source support,
provenance and intent bindings are computed here from those rows, never typed by hand.

Structure follows what the repository records about the historical values (no historical text survives):
- intent cards (docs/receipts/knowledge-index-design-v1.json, docs/receipts/knowledge/slurp-intent-contrast-cards-v1.json):
  one activation line, exactly one named neighbour (the receipt's), one uncertainty sentence shared by all 60 cards,
  one shared leading limitation and a card-specific limitation on exactly the 14 cards the receipt names;
- slot cards (2026-09-13 slot root review, 2026-09-17 revisions review): one value per entity type with a role per
  observed intent pair, extraction cues, admissible formats, one shared missing-behaviour sentence in v1, ambiguity,
  a shared leading limitation and card-specific limitations; v2 changes exactly eight cards as the revision review
  section 3 describes and leaves every binding, role, ambiguity entry and limitation unchanged.

usage: python3 -I author_semantic_knowledge_npu_port_v1.py <data-root> <repo-root> <out-dir>
Writes create-only; refuses to overwrite any existing byte.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

SOURCE_SCHEMA = "saea-authored-semantic-knowledge-source-v1"
INTENT_POLICY = "saea-authored-intent-boundaries-v1"
SLOT_POLICY = "saea-authored-slot-contracts-v1"
SUPPORT_RELPATH = "prepared/SAEA-SLURP-SEMANTIC-AUTHORING-PREPARE-V1/support.json"
INTENT_ROSTER = "prepared/SAEA-SLURP-SEMANTIC-AUTHORING-PREPARE-V1/intent-support-construction-roster.json"
SLOT_ROSTER = "prepared/SAEA-SLURP-SEMANTIC-AUTHORING-PREPARE-V1/slot-support-construction-roster.json"
CONTRAST_RECEIPT = "docs/receipts/knowledge/slurp-intent-contrast-cards-v1.json"

# --- intent boundaries --------------------------------------------------------------------------------------------

INTENT_UNCERTAINTY = (
    "Treat this boundary as candidate guidance: when the evidence fits it only in part, keep it beside its named "
    "neighbor, decide by the requested operation and its object, and never reject it because a usual slot is absent."
)
INTENT_SHARED_LIMITATION = (
    "Written from at most eight selected TRAIN requests for this intent; it is not an exhaustive definition, a "
    "lexical lookup or a required-slot schema, and it does not eliminate other labels."
)

# intent_id: (activation boundary, neighbouring intent, distinction, card-specific limitation or None)
INTENTS = {
    "alarm_query": (
        "The user asks which alarms exist, when they ring or what they are for.",
        "alarm_set",
        "asking whether, when or why an alarm is set inspects existing alarms; alarm_set creates or moves one.",
        None),
    "alarm_remove": (
        "The user wants an existing alarm or recurring wake-up call deleted, cancelled or disabled.",
        "alarm_set",
        "removal ends an alarm, including a request not to be woken; naming a new ring time without cancelling "
        "anything is alarm_set.",
        None),
    "alarm_set": (
        "The user wants a new alarm created or an existing alarm moved to a ring time.",
        "calendar_set",
        "an alarm rings at a clock time or after a delay; recording an event, meeting or reminder in the calendar "
        "is calendar_set.",
        None),
    "audio_volume_down": (
        "The user wants the output volume lowered, or complains that it is too loud.",
        "audio_volume_mute",
        "lowering keeps sound audible, by an amount or to a level; asking for silence, quiet or no sound is "
        "audio_volume_mute.",
        None),
    "audio_volume_mute": (
        "The user wants the speaker silenced or muted, now or for a stated duration.",
        "audio_volume_down",
        "mute removes sound entirely, including a level of zero; a request that only reduces loudness is "
        "audio_volume_down.",
        "Two selected requests are a bare stop and a bare call for quiet; such short commands could also end "
        "playback, so they show breadth, not a rule that every stop is a mute."),
    "audio_volume_other": (
        "The user asks to adjust or open volume settings without a clear direction.",
        "audio_volume_down",
        "audio_volume_other covers undirected changes, settings and sound quality; a request that names lower or quieter is "
        "audio_volume_down.",
        None),
    "audio_volume_up": (
        "The user wants the volume raised or asks the assistant to speak louder.",
        "audio_volume_other",
        "audio_volume_up names an increase, louder speech or a higher level; a volume change with no stated direction is "
        "audio_volume_other.",
        None),
    "calendar_query": (
        "The user asks about their own scheduled events, meetings, reminders or appointments.",
        "recommendation_events",
        "calendar_query inspects the user's own schedule; asking what public events are happening in an area is "
        "recommendation_events.",
        None),
    "calendar_remove": (
        "The user wants one or more calendar events, appointments or schedules deleted.",
        "alarm_remove",
        "calendar_remove deletes events or appointments from the calendar; cancelling an alarm or a wake-up call "
        "is alarm_remove.",
        None),
    "calendar_set": (
        "The user wants an event, meeting, reminder or recurrence added to the calendar.",
        "calendar_query",
        "calendar_set creates or schedules an entry, often with a time or a person; asking what is already "
        "scheduled is calendar_query.",
        None),
    "cooking_query": (
        "The user asks a cooking question such as an ingredient swap or the quickest method.",
        "cooking_recipe",
        "support for both labels asks how to prepare food; cooking_query leans to swaps and method questions, "
        "cooking_recipe to a named dish.",
        "Only three selected support rows exist and one of them asks for a recipe; the boundary with "
        "cooking_recipe overlaps, so keep both candidates open when a request fits either."),
    "cooking_recipe": (
        "The user asks how to cook or prepare a dish, or for a recipe or a cooking time.",
        "cooking_query",
        "cooking_recipe usually names the dish to prepare; a question about swapping an ingredient within a dish "
        "may be cooking_query.",
        "Selected support also asks for a substitute and whether a dish comes from a country; these overlap "
        "cooking_query and factual questions and do not make every food question a recipe."),
    "datetime_convert": (
        "The user wants a time converted between zones or places, or a time zone changed.",
        "datetime_query",
        "datetime_convert relates a stated time or zone to another one; asking only the current time or date in one place "
        "is datetime_query.",
        None),
    "datetime_query": (
        "The user asks the current time or date, a day of the week, or when a date falls.",
        "datetime_convert",
        "datetime_query asks one time or date, possibly for a named place; translating a stated time into another "
        "zone is datetime_convert.",
        None),
    "email_addcontact": (
        "The user wants an email address or a new contact saved to the contact list.",
        "email_sendemail",
        "email_addcontact stores an address or a contact; a request that composes, sends or replies to a message is "
        "email_sendemail.",
        None),
    "email_query": (
        "The user asks whether new emails arrived, from whom, or what an email says.",
        "email_querycontact",
        "email_query inspects received mail, often by sender or day; asking for a contact's details or number is "
        "email_querycontact.",
        None),
    "email_querycontact": (
        "The user asks about a stored contact, such as a person's details or number.",
        "email_addcontact",
        "email_querycontact looks up or selects existing contact information; saving a new address or a new contact is "
        "email_addcontact.",
        "Selected support also adds a person to favorite contacts, checks for a received message and contains a "
        "one-word fragment; these overlap neighboring email labels and do not define the boundary."),
    "email_sendemail": (
        "The user wants an email composed, sent, replied to or mailed to someone.",
        "social_post",
        "email_sendemail addresses a message by email or mail to a recipient; publishing to a feed or tweeting at a "
        "company is social_post.",
        None),
    "general_greet": (
        "The user greets the assistant or asks how it is doing.",
        "datetime_query",
        "greet support is a greeting or a question about the assistant's state; a request whose content asks the "
        "date or the time is datetime_query.",
        "One selected greeting continues with a question about today and carries a date annotation; a greeting "
        "followed by a real request may belong to that request's label."),
    "general_joke": (
        "The user asks for a joke or a kind of joke, or starts telling one.",
        "general_quirky",
        "joke support asks for or sets up a joke; an unusual or playful request does not become a joke without "
        "that request.",
        None),
    "general_quirky": (
        "The user makes open conversation, opinions or musings with no actionable target.",
        "qa_factoid",
        "a question asking a determinate fact about a named person, place or thing is qa_factoid; avoid treating "
        "all questions as quirky.",
        "One selected request schedules coffee at a clock time and carries drink, time and date annotations; it "
        "records heterogeneous support, so do not use this label as an automatic catch-all."),
    "iot_cleaning": (
        "The user wants a vacuum or cleaning robot started, or asks about one.",
        "iot_wemo_on",
        "iot_cleaning names a vacuum or the cleaning of a room; switching on a plug or socket is iot_wemo_on even when "
        "a device is attached to it.",
        None),
    "iot_coffee": (
        "The user wants coffee made, or the coffee machine started.",
        "takeaway_order",
        "iot_coffee asks the home machine or the assistant for a drink; ordering food from a restaurant or shop is "
        "takeaway_order.",
        "One selected request orders coffee from a named coffee shop, which crosses the device and ordering "
        "distinction; do not treat every coffee order as a machine request, or the reverse."),
    "iot_hue_lightchange": (
        "The user wants the color of the lights changed, by name or relatively.",
        "iot_hue_lightdim",
        "iot_hue_lightchange alters color or shade; making the lights darker or reducing their brightness is "
        "iot_hue_lightdim.",
        "One selected request sets the lights to a percentage, which is a brightness level rather than a color; it "
        "records the support, not a rule that levels belong here."),
    "iot_hue_lightdim": (
        "The user wants the lights dimmed or their brightness reduced without turning them off.",
        "iot_hue_lightoff",
        "dimming keeps the lights on at a lower level; switching them off entirely is iot_hue_lightoff.",
        None),
    "iot_hue_lightoff": (
        "The user wants lights switched off, in general or in a named room.",
        "iot_wemo_off",
        "iot_hue_lightoff names lights or a lamp; turning off a smart plug or socket is iot_wemo_off, even when a light is "
        "plugged into it.",
        None),
    "iot_hue_lighton": (
        "The user wants lights switched on, or says it is too dark to see.",
        "iot_hue_lightup",
        "iot_hue_lighton starts lights that are off; raising the brightness of lights that are already on is "
        "iot_hue_lightup.",
        None),
    "iot_hue_lightup": (
        "The user wants the lights brighter or their brightness increased.",
        "iot_hue_lighton",
        "iot_hue_lightup raises the brightness of lit lights; a request to switch lights on, or a complaint that it is "
        "dark, is iot_hue_lighton.",
        None),
    "iot_wemo_off": (
        "The user wants a smart plug or socket switched off or kept inactive.",
        "iot_hue_lightoff",
        "iot_wemo_off names a plug, a socket or a wemo device; switching off lights or a lamp is iot_hue_lightoff.",
        None),
    "iot_wemo_on": (
        "The user wants a smart plug or socket switched on.",
        "iot_hue_lighton",
        "iot_wemo_on names a plug, a socket or a wemo device; switching on lights or a lamp is iot_hue_lighton.",
        None),
    "lists_createoradd": (
        "The user wants a new list created or an item added to a list.",
        "lists_query",
        "lists_createoradd changes list contents by making a list or adding to one; reading or checking what a list holds "
        "is lists_query.",
        None),
    "lists_query": (
        "The user asks to see lists, what a list holds, or whether an item is on it.",
        "lists_createoradd",
        "lists_query reads lists or checks for an item; creating a list or adding an item to it is lists_createoradd.",
        None),
    "lists_remove": (
        "The user wants an item removed from a list, or a whole list deleted.",
        "calendar_remove",
        "lists_remove deletes list items or lists; deleting a calendar event or an appointment from the calendar is "
        "calendar_remove.",
        None),
    "music_dislikeness": (
        "The user says they dislike the current song or a kind of music.",
        "audio_volume_down",
        "music_dislikeness records a negative preference about the music; a complaint that the sound is too loud asks "
        "for less volume.",
        None),
    "music_likeness": (
        "The user says they like a song, artist or genre, or asks to save it.",
        "play_music",
        "music_likeness records or saves a preference; a request to start playing music, even a favorite, is play_music.",
        "Selected support includes an unclear request about a music file and a request to save preferences on a "
        "playlist; these overlap neighboring music labels and do not define the boundary."),
    "music_query": (
        "The user asks about current or past music, such as the song, singer or playlist.",
        "play_music",
        "music_query asks for information about music; a request to play or start it is play_music.",
        None),
    "music_settings": (
        "The user wants the playback mode changed, such as shuffle or repeat.",
        "play_music",
        "music_settings changes how music plays; choosing what to play is play_music, even when shuffle is mentioned.",
        "One selected request asks to show the current track in a music player, which is closer to a question "
        "about the music; it records the support, not a rule."),
    "news_query": (
        "The user asks for news, headlines or updates on a topic, place or person.",
        "qa_factoid",
        "news_query asks what is new or current; a question for a stable fact about a person or a place is qa_factoid.",
        None),
    "play_audiobook": (
        "The user wants an audiobook or a book played, resumed or restarted.",
        "play_music",
        "play_audiobook names a book, an audiobook or reading; playing songs or an artist is play_music, although the "
        "support also mixes songs with audiobooks.",
        None),
    "play_game": (
        "The user wants to play a game, often a named one, with the assistant.",
        "play_music",
        "play_game names a game or playing together; play followed by a song, an artist or a genre is play_music.",
        None),
    "play_music": (
        "The user wants music played, by song, artist, genre or playlist.",
        "music_settings",
        "play_music selects what to play; changing only shuffle or repeat for the current playback is "
        "music_settings.",
        None),
    "play_podcasts": (
        "The user wants a podcast or an episode played, or the next episode.",
        "play_audiobook",
        "play_podcasts names a podcast, a show or an episode; continuing a named book or an audiobook is "
        "play_audiobook.",
        None),
    "play_radio": (
        "The user wants the radio, a station or a radio show turned on.",
        "play_music",
        "play_radio names a radio, a station or a show; playing songs, artists or playlists is play_music.",
        None),
    "qa_currency": (
        "The user asks an exchange rate or the value of one currency in another.",
        "qa_stock",
        "qa_currency compares units of money; asking the price or movement of a company's stock is qa_stock.",
        None),
    "qa_definition": (
        "The user asks the meaning or the definition of a word or a term.",
        "qa_factoid",
        "qa_definition asks what a term means; asking a fact about a named entity is qa_factoid.",
        None),
    "qa_factoid": (
        "The user asks a factual question about a person, place, thing or event.",
        "qa_definition",
        "qa_factoid asks a fact such as an age, a year or a location; asking what a word means is qa_definition.",
        None),
    "qa_maths": (
        "The user asks for an arithmetic result or a calculation.",
        "qa_currency",
        "qa_maths computes with numbers; converting an amount between currencies is qa_currency.",
        None),
    "qa_stock": (
        "The user asks a company's stock price, its rate or market information.",
        "qa_currency",
        "qa_stock names a company or the market; the strength of one currency against another is qa_currency, though "
        "the support also asks a currency's value here.",
        None),
    "recommendation_events": (
        "The user asks what events or activities are happening in an area or at a time.",
        "calendar_query",
        "recommendation_events discovers public happenings nearby; asking about the user's own scheduled events is "
        "calendar_query.",
        None),
    "recommendation_locations": (
        "The user asks for a recommended place, such as a restaurant, bar or shop nearby.",
        "transport_query",
        "recommendation_locations recommends where to go; asking for directions or the distance to a known place is "
        "transport_query.",
        "Selected support includes a two-word price question and a bare request for nearby places; these short "
        "requests overlap other labels and do not define the boundary."),
    "recommendation_movies": (
        "The user asks for movie suggestions, reviews or what is playing.",
        "recommendation_locations",
        "recommendation_movies recommends films or showings; asking only for a nearby place such as a restaurant is "
        "recommendation_locations.",
        "Selected support includes a request for the nearest theatre and a request mixing movies and songs; these "
        "overlap neighboring labels and do not define the boundary."),
    "social_post": (
        "The user wants something posted, tweeted or sent as a complaint on social media.",
        "social_query",
        "social_post publishes content or a complaint; checking what others have posted is social_query.",
        None),
    "social_query": (
        "The user asks what is new on social media, such as posts, likes or mentions.",
        "social_post",
        "social_query reads social media activity; publishing a post or a tweet is social_post.",
        None),
    "takeaway_order": (
        "The user wants food ordered for delivery or takeout, or asks for order options.",
        "takeaway_query",
        "takeaway_order places or browses an order; checking delivery availability or the status of an order is "
        "takeaway_query.",
        "One selected request carries a business-name annotation that does not occur in its words; do not copy "
        "entity values from inconsistent support."),
    "takeaway_query": (
        "The user asks about an order's status or whether a place offers delivery or takeaway.",
        "takeaway_order",
        "takeaway_query checks status or availability; placing the order itself is takeaway_order.",
        "Selected support includes a restaurant recommendation for take out and a delivery-time question; these "
        "overlap neighboring labels and do not define the boundary."),
    "transport_query": (
        "The user asks for directions, a distance, a route or train times.",
        "transport_ticket",
        "transport_query seeks route or timetable information; booking or reserving a ticket or a seat is "
        "transport_ticket.",
        None),
    "transport_taxi": (
        "The user wants a taxi, cab, ride or ride-share booked or found.",
        "transport_ticket",
        "transport_taxi requests a car ride to or from a place; booking a train ticket or seat is transport_ticket.",
        None),
    "transport_ticket": (
        "The user wants a train ticket or seat booked, reserved or searched for.",
        "transport_query",
        "transport_ticket acquires travel; asking train times or directions without booking anything is transport_query.",
        None),
    "transport_traffic": (
        "The user asks about traffic, congestion, jams or roadblocks.",
        "transport_query",
        "transport_traffic asks about road conditions; asking a route, directions or train times is transport_query.",
        None),
    "weather_query": (
        "The user asks about the weather, the temperature or whether to bring weather gear.",
        "datetime_query",
        "weather_query asks about conditions, even when a date is named; asking only the date or the time is "
        "datetime_query.",
        None),
}

# --- slot contracts -----------------------------------------------------------------------------------------------

SLOT_MISSING = (
    "When no span of the utterance expresses this slot, omit it; never invent a value from the intent, a default, "
    "the user profile or a frequent pairing."
)
SLOT_SHARED_LIMITATION = (
    "Bindings index at most three selected TRAIN examples per intent pair; they guide extraction and are not "
    "required fields, frequency rules or an exhaustive list of legal types."
)

# slot_id: dict(roles={intent: role}, cues=[...], formats=[...], ambiguity=str, limitation=str)
SLOTS = {
    "alarm_type": dict(
        roles={
            "alarm_query": "the kind of alarm whose setting is checked",
            "alarm_remove": "the kind of alarm or wake-up call to cancel",
            "alarm_set": "the kind of alarm to create, such as a wake-up",
        },
        cues=["A short phrase naming the alarm's kind or purpose, often wake up, before alarm, alert or calls.",
              "Keep the ring time, the date and the part of day in their own slots."],
        formats=["The spoken kind words, two words in the support (wake up, house alarm)."],
        ambiguity="A purpose such as work or a flight can be an event name rather than an alarm kind; use "
                  "alarm_type only for a stated category of alarm.",
        limitation="Seven mentions over three intents, five of them the same phrase; the support is too narrow to "
                   "define other alarm kinds."),
    "app_name": dict(
        roles={
            "calendar_remove": "the calendar application holding the event",
            "cooking_recipe": "the cooking application to open or download",
            "music_likeness": "the music service where a liked item is saved",
            "music_settings": "the music service whose playback mode changes",
            "play_audiobook": "the audiobook service to play from",
            "play_music": "the music service to play from",
            "play_radio": "the streaming or radio service to tune",
            "takeaway_order": "the food-ordering service to order through",
            "transport_query": "the navigation application to use",
            "transport_taxi": "the ride application to open",
        },
        cues=["A named application or service, often after on, from, in or open.",
              "Separate the application from the content it plays and from the device that runs it."],
        formats=["The spoken service name, one or two words; a generic word such as app stays when spoken with it."],
        ambiguity="A service name can also be a station, a company or a media channel; choose app_name when it "
                  "names the software or service used to act.",
        limitation="Twenty mentions of a few streaming and ordering services; a ride company under transport_taxi "
                   "is labeled transport_agency in other support, so the role can shift by intent."),
    "artist_name": dict(
        roles={
            "email_sendemail": "a performer named as the email recipient",
            "general_quirky": "a performer the user muses or asks about",
            "music_likeness": "the performer whose music is liked or saved",
            "music_query": "the performer asked about or compared",
            "music_settings": "the performer whose songs are shuffled or repeated",
            "play_audiobook": "the performer named with an audiobook request",
            "play_music": "the performer whose music should play, often after by",
            "qa_factoid": "the performer a fact is asked about",
        },
        cues=["A performer or band name, often after by or of, or as the subject of a question.",
              "A song title next to the name is song_name; keep the two spans apart."],
        formats=["The spoken name, one or two words, with a leading article when spoken as part of the name."],
        ambiguity="Support labels some named people as person instead; use artist_name when the name is presented "
                  "as a performer of music, otherwise person.",
        limitation="Assignments between artist_name and person are inconsistent in the support, and some annotated "
                   "spellings differ from the spoken words; do not copy them as a lookup."),
    "audiobook_author": dict(
        roles={"play_audiobook": "the author whose book should play"},
        cues=["An author's name, after by or before book, in a request to play a book."],
        formats=["The spoken full or short name, two words in the support."],
        ambiguity="A name after by in a book request is the author; a name attached to songs is artist_name.",
        limitation="Three mentions under one intent; the support is too small for general rules."),
    "audiobook_name": dict(
        roles={"play_audiobook": "the title of the book to play, resume or restart"},
        cues=["The book title after play, read, continue or resume, or after the audiobook of.",
              "Playback words such as resume playing belong to player_setting, not to the title."],
        formats=["The spoken title, one to five words, with a leading article only when spoken as part of it."],
        ambiguity="A title can contain numbers or names; keep it whole and do not split it into date, person or "
                  "other slots.",
        limitation="Eight mentions under one intent; one title keeps its leading article and another drops it, so "
                   "the article convention is not fixed."),
    "business_name": dict(
        roles={
            "calendar_query": "the organization the scheduled meeting is with",
            "calendar_set": "the named venue where the event takes place",
            "cooking_recipe": "the website or brand the recipe comes from",
            "email_query": "the organization that sent the email",
            "email_sendemail": "the organization the email is sent to",
            "iot_coffee": "the named coffee shop to order from",
            "lists_query": "the brand whose items the list covers",
            "qa_stock": "the company whose stock is asked about",
            "recommendation_events": "the named venue asked about",
            "recommendation_locations": "the named business to locate",
            "social_post": "the company a post or complaint addresses",
            "takeaway_order": "the restaurant or chain to order from",
            "takeaway_query": "the restaurant whose delivery is checked",
            "transport_query": "the named destination business",
            "transport_taxi": "the named business the ride goes to",
        },
        cues=["A proper name of a company, shop, chain or organization, often after from, at, to or of.",
              "A category noun after the name, such as restaurant or coffee shop, is business_type."],
        formats=["The spoken name as uttered, usually one or two words; a spoken web address keeps the name words."],
        ambiguity="A company name can also be a social medium, an application or a stock name; choose business_name "
                  "when it names the organization acted on or asked about.",
        limitation="One takeaway example carries a business-name annotation that does not occur in its words, and "
                   "some names merge spoken domain words; do not copy such values."),
    "business_type": dict(
        roles={
            "general_joke": "a kind of place named in a joke",
            "general_quirky": "a kind of place mentioned in conversation",
            "iot_coffee": "the kind of shop to order coffee from",
            "recommendation_events": "the kind of venue or gathering sought",
            "recommendation_locations": "the kind of place to recommend",
            "recommendation_movies": "the kind of venue showing films",
            "social_post": "the kind of business a complaint targets",
            "takeaway_order": "the kind of place to order from",
            "takeaway_query": "the kind of place whose service is checked",
            "transport_query": "the kind of place to get directions to",
            "transport_taxi": "the kind of place the ride goes to",
        },
        cues=["A category noun for a place or business, such as restaurant, bar or theater.",
              "A cuisine before the noun is food_type; keep it out of the category span."],
        formats=["The spoken category noun, singular or plural as spoken."],
        ambiguity="A named chain is business_name and a category is business_type; a word like show can be an "
                  "event rather than a venue.",
        limitation="One selected span names a chain as a category and one span keeps a following modifier inside "
                   "it; treat these as annotation noise."),
    "change_amount": dict(
        roles={
            "audio_volume_down": "the lower volume target or the reduction",
            "audio_volume_mute": "a zero level that silences the output",
            "audio_volume_other": "an unspecified or average level",
            "audio_volume_up": "the higher volume target or the increase",
            "iot_hue_lightchange": "the light level to set",
            "iot_hue_lightdim": "the dimmer level or the reduction",
            "iot_hue_lightup": "the brighter level or the increase",
        },
        cues=["A level or change with its relation word: to, by or at plus a number, a percentage or a word such "
              "as half.",
              "A target level introduced by to differs from a relative change introduced by by; keep the relation "
              "word in the span."],
        formats=["The spoken phrase with to, by or at and numbers in words (to ten, by twenty percent)."],
        ambiguity="A vague amount such as a bit is still an amount; do not convert it into a number or a direction.",
        limitation="One selected request says lower but is labeled as a volume increase, and one bare word is "
                   "labeled an amount; rely on the spoken relation, not on these labels."),
    "coffee_type": dict(
        roles={"iot_coffee": "the kind of coffee drink to make"},
        cues=["A coffee drink name or the user's usual choice, after make, like or a cup of."],
        formats=["One spoken word as uttered."],
        ambiguity="Coffee alone is the requested drink, not a coffee_type, unless a kind is named.",
        limitation="Four mentions under one intent, and one annotated spelling differs from the spoken word; do not "
                   "copy spellings."),
    "color_type": dict(
        roles={
            "iot_hue_lightchange": "the target color or shade of the lights",
            "iot_hue_lightdim": "the color that accompanies a dimming request",
            "iot_hue_lightoff": "a color named beside a lights-off request",
        },
        cues=["A color name or a relative shade word such as softer, after to or before lights.",
              "When two colors are spoken, keep each as its own mention."],
        formats=["The spoken color words, one or two words (red, dull blue)."],
        ambiguity="A switch between colors names both the old and the new color; record each spoken color without "
                  "deciding which is the target.",
        limitation="Seven mentions; one request combines lights off in one room with a color change in another, so "
                   "the intent need not match the color mention."),
    "cooking_type": dict(
        roles={"cooking_recipe": "the cooking method asked about"},
        cues=["A cooking verb such as roll, sear, steam or cook, followed by the food."],
        formats=["One spoken verb as uttered."],
        ambiguity="A generic cook can be the method or just the request verb; annotate it when it names how the "
                  "food is prepared.",
        limitation="Four mentions under one intent; the support does not show whether every cooking verb is "
                   "annotated."),
    "currency_name": dict(
        roles={
            "qa_currency": "a currency in the rate or conversion",
            "qa_stock": "a currency whose value is asked about",
            "transport_ticket": "the currency of a ticket budget",
        },
        cues=["A currency name or spelled code, often in pairs joined by to, per, for or versus.",
              "Keep each currency as its own mention, in spoken order."],
        formats=["The spoken name or spelled letters as uttered, one to three words (u. s. d.)."],
        ambiguity="In a conversion the order of the two currencies carries meaning; do not merge or reorder "
                  "distinct mentions solely because their types match.",
        limitation="Thirteen mentions; one spoken currency in a conversion is left unannotated and one annotated "
                   "name differs from the spoken one; do not copy either."),
    "date": dict(
        roles={
            "alarm_query": "the day whose alarms are checked",
            "alarm_remove": "the day or days of the alarm to cancel",
            "alarm_set": "the day the alarm should ring",
            "calendar_query": "the day of the schedule asked about",
            "calendar_remove": "the day of the event to delete",
            "calendar_set": "the day of the event or reminder",
            "cooking_recipe": "a year that bounds the recipe",
            "datetime_query": "the day or date asked about",
            "email_query": "the day the emails arrived",
            "email_sendemail": "a day named in the email content",
            "general_greet": "a day named within a greeting",
            "general_joke": "the day the jokes are for",
            "general_quirky": "a day mentioned in conversation",
            "iot_cleaning": "the day the cleaning should run",
            "iot_coffee": "the day or recurrence for coffee",
            "lists_createoradd": "the day the list is for",
            "lists_query": "the day whose list is checked",
            "lists_remove": "the day of the list or item to remove",
            "music_likeness": "the period of the liked items",
            "music_query": "the period of the music asked about",
            "news_query": "the day of the news",
            "play_audiobook": "the day playback last happened",
            "play_music": "a year bounding the music to play",
            "play_podcasts": "the day of the episode",
            "play_radio": "the day of the show or the listening",
            "qa_currency": "the day of the exchange rate",
            "qa_factoid": "the period the fact concerns",
            "qa_maths": "the period in a calculation",
            "qa_stock": "the day of the stock price",
            "recommendation_events": "the day or weekend of the events",
            "recommendation_locations": "the day of the visit",
            "recommendation_movies": "the week or weekend of showings",
            "social_post": "a day named in the post",
            "social_query": "the day of the social activity",
            "takeaway_order": "the day of the deal or order",
            "transport_query": "the day of travel asked about",
            "transport_taxi": "the day of the ride",
            "transport_ticket": "the day of travel to book",
            "transport_traffic": "the day of the traffic",
            "weather_query": "the day of the forecast",
        },
        cues=["A calendar expression: a day name, a relative day, a week, month or year, or a spoken ordinal date.",
              "A clock reading is time and a part of day is timeofday; split them from the date when spoken "
              "together."],
        formats=["The spoken words as uttered, numbers in words and a possessive kept as spoken; no normalized "
                 "calendar value."],
        ambiguity="A month or year inside a title need not be a date, and a duration such as next two days may be "
                  "time; prefer date for a calendar anchor.",
        limitation="Temporal modifiers are omitted or typed differently across the support (a spoken next can fall "
                   "outside the span, a weekend can be time); do not reproduce these as rules."),
    "definition_word": dict(
        roles={"qa_definition": "the term whose meaning is asked"},
        cues=["The word or phrase after define, meaning of, definition of or what's a."],
        formats=["The spoken term as uttered, with a leading article only when spoken inside the term."],
        ambiguity="After tell me about, the term can be a definition word or the subject of a fact; choose by "
                  "whether a meaning is asked.",
        limitation="Three mentions under one intent; one span keeps its article and the others do not, so the "
                   "article convention is not fixed."),
    "device_type": dict(
        roles={
            "alarm_query": "the device holding the alarms",
            "audio_volume_down": "the device whose output is lowered",
            "audio_volume_mute": "the speakers to mute",
            "audio_volume_up": "the player whose volume rises",
            "email_querycontact": "the contact store consulted",
            "general_quirky": "a device mentioned in conversation",
            "iot_cleaning": "the vacuum or robot to start",
            "iot_coffee": "the coffee appliance to start",
            "iot_hue_lightdim": "the light fixture to dim",
            "iot_hue_lightoff": "the lights or fixture to switch off",
            "iot_hue_lighton": "the lights or lamp to switch on",
            "iot_wemo_off": "the plug or socket to switch off",
            "iot_wemo_on": "the plug or socket to switch on",
            "lists_query": "the screen to show the lists on",
            "music_settings": "the music player to adjust",
            "play_game": "the device to play the game on",
            "play_music": "the music player to start",
            "play_radio": "the radio set to play on",
            "social_post": "the device holding the content to post",
        },
        cues=["A device or appliance noun, possibly with a brand or room word spoken inside it.",
              "A room alone is house_place and a service is app_name; keep the device span to the equipment."],
        formats=["The spoken device words, one to three words, singular or plural as spoken."],
        ambiguity="A room word can be inside the device span or a separate house_place; follow the spoken grouping "
                  "and do not split a compound device name.",
        limitation="Room words are sometimes inside the device span and sometimes separate, and generic lights are "
                   "annotated only in some requests; do not treat either as a rule."),
    "drink_type": dict(
        roles={
            "cooking_recipe": "the drink to make or the one used",
            "general_quirky": "a drink mentioned in conversation",
            "lists_createoradd": "the drink to add to a list",
            "recommendation_locations": "the drink a shop should sell",
        },
        cues=["A beverage noun, such as coffee, tea, beer or water."],
        formats=["The spoken drink words, one or two words."],
        ambiguity="In a recipe a drink can also be a food ingredient; follow whether it is consumed as a drink.",
        limitation="Six mentions; one request labels milk as a drink and buttermilk as a food, so the boundary with "
                   "food_type is not fixed."),
    "email_address": dict(
        roles={
            "email_addcontact": "the address to store for a contact",
            "email_sendemail": "the address to send the email to",
        },
        cues=["A spoken address with at and dot words, such as a name at a domain dot com."],
        formats=["The spoken address with the at sign and dot com as spoken, three words in the support."],
        ambiguity="A person's name next to the address is person; keep the address span to the address itself.",
        limitation="Eight mentions over two intents, all with common domains; do not infer a domain that was not "
                   "spoken."),
    "email_folder": dict(
        roles={
            "email_query": "the mailbox folder to check",
            "email_sendemail": "the folder holding the email to send",
        },
        cues=["A mailbox folder word such as inbox or drafts."],
        formats=["One spoken word, without the word folder."],
        ambiguity="Email or mail alone is not a folder.",
        limitation="Four mentions over two intents; too few to define other folder names."),
    "event_name": dict(
        roles={
            "alarm_query": "the event the alarm is for",
            "alarm_remove": "the named alarm or occasion to cancel",
            "alarm_set": "the event or purpose the alarm serves",
            "calendar_query": "the scheduled event asked about",
            "calendar_remove": "the event to delete",
            "calendar_set": "the event or reminder to schedule",
            "datetime_query": "the holiday or event whose date is asked",
            "email_query": "the occasion the emails concern",
            "email_querycontact": "the occasion of a contact, such as a birthday",
            "email_sendemail": "the occasion the email concerns",
            "general_quirky": "an occasion mentioned in conversation",
            "lists_createoradd": "the occasion a list or list item is for",
            "lists_remove": "the occasion of the list to delete",
            "qa_factoid": "the occasion a fact is asked about",
            "recommendation_events": "the kind of event sought",
            "recommendation_movies": "the screening or film event asked about",
            "social_post": "the occasion a post celebrates",
            "social_query": "the life event asked about",
            "transport_taxi": "the event the ride follows",
            "transport_traffic": "the traffic incident asked about",
            "weather_query": "the occasion the weather affects",
        },
        cues=["The occasion or activity itself, such as meeting, appointment, birthday or concert.",
              "A person, relation, place or time attached to the event goes to its own slot."],
        formats=["The spoken occasion words as uttered, singular or plural as spoken."],
        ambiguity="An activity phrase can also be reminder text; keep only the occasion and leave the people, "
                  "places and times to their slots.",
        limitation="Some spans cover whole clauses or a counted repetition, and a festival or title can look like "
                   "an event; do not copy those spans as a rule."),
    "food_type": dict(
        roles={
            "cooking_query": "the ingredient or dish in the question",
            "cooking_recipe": "the dish or food to prepare",
            "general_joke": "the food a joke is about",
            "general_quirky": "a food mentioned in conversation",
            "qa_factoid": "the food a fact is asked about",
            "recommendation_locations": "the food or cuisine to find",
            "takeaway_order": "the dish or cuisine to order",
            "takeaway_query": "the food whose delivery is checked",
            "transport_query": "the cuisine of the destination",
        },
        cues=["A dish, food or cuisine word, including a cuisine adjective before restaurant.",
              "Keep the restaurant noun as business_type and a cooking verb as cooking_type."],
        formats=["The spoken food words, one to five words, keeping spoken modifiers (four cheese omelette)."],
        ambiguity="In a substitution question both foods are separate mentions; do not merge them or decide which "
                  "one is the target.",
        limitation="The boundary with ingredient and drink_type is not consistent in the support, and one joke "
                   "span covers a whole phrase; do not reproduce these as rules."),
    "game_name": dict(
        roles={"play_game": "the game to play"},
        cues=["A named game after play, a game of, or in."],
        formats=["The spoken name, one or two words."],
        ambiguity="A sports match on a device can be annotated as a game name; keep it when it is the thing to "
                  "play.",
        limitation="Four mentions under one intent; spellings follow the spoken words, not official titles."),
    "game_type": dict(
        roles={"play_game": "the kind of game to play"},
        cues=["A category word for games, such as video before games."],
        formats=["One spoken word."],
        ambiguity="A named game is game_name; game_type is only a category.",
        limitation="One mention in the whole support; it cannot support general rules."),
    "general_frequency": dict(
        roles={
            "alarm_set": "how often the alarm repeats",
            "calendar_query": "the recurrence of the event asked about",
            "calendar_remove": "the recurrence of the events to delete",
            "calendar_set": "how often the event or reminder repeats",
            "email_sendemail": "how often the reply rule applies",
            "news_query": "how often news updates arrive",
            "qa_maths": "the interval in a calculation",
            "weather_query": "the period of the weather report",
        },
        cues=["A recurrence word or phrase, such as everyday, weekly, monthly, annual, recurring or seven days a "
              "week.",
              "A weekday inside every friday stays in the frequency span."],
        formats=["The spoken recurrence words, one to four words."],
        ambiguity="A repeating weekday attached to a date can be annotated as date instead; prefer "
                  "general_frequency for repetition.",
        limitation="Fourteen mentions; the boundary with date for repeating weekdays is not consistent, so do not "
                   "reproduce it as a rule."),
    "house_place": dict(
        roles={
            "alarm_query": "the home whose alarms are checked",
            "calendar_set": "the room where the meeting is held",
            "iot_cleaning": "the room or house to clean",
            "iot_hue_lightchange": "the room whose light color changes",
            "iot_hue_lightdim": "the room whose lights dim",
            "iot_hue_lightoff": "the room whose lights go off",
            "iot_hue_lighton": "the room or spot whose lights go on",
            "iot_hue_lightup": "the room whose lights brighten",
            "iot_wemo_off": "the room of the plug to switch off",
            "iot_wemo_on": "the room of the plug to switch on",
            "play_radio": "the room where the radio plays",
        },
        cues=["A room or part of the home, such as living room, kitchen or bedroom, after in or before a device.",
              "An outside place or a city is place_name."],
        formats=["The spoken room words, one or two words, singular or plural as spoken."],
        ambiguity="A room before a device noun can be part of the device span; when two rooms are spoken, keep "
                  "each as its own mention.",
        limitation="One span keeps an article and the others do not, and a spot word is labeled a room; the "
                   "article and scope conventions are not fixed."),
    "ingredient": dict(
        roles={"cooking_recipe": "the ingredient to cook or replace"},
        cues=["An ingredient noun in a cooking question, such as rice."],
        formats=["The spoken words, one or two words."],
        ambiguity="The same food can be food_type elsewhere; ingredient appears for components rather than "
                  "finished dishes, but not consistently.",
        limitation="Three mentions under one intent; the support does not separate ingredient from food_type "
                   "reliably."),
    "joke_type": dict(
        roles={"general_joke": "the kind or subject of joke wanted"},
        cues=["A word for the kind or subject of the joke, such as funny, or what it is about."],
        formats=["The spoken word or short phrase."],
        ambiguity="A person or relation the joke is about can be its own slot; keep joke_type to the kind or "
                  "theme.",
        limitation="Six mentions under one intent; one span keeps the word about and the others do not."),
    "list_name": dict(
        roles={
            "calendar_query": "the schedule or task list asked about",
            "calendar_remove": "the schedule to clear",
            "email_addcontact": "the contact list to add to",
            "email_querycontact": "the contact list to open",
            "lists_createoradd": "the list to create or add to",
            "lists_query": "the list whose contents are asked",
            "lists_remove": "the list to delete or remove from",
        },
        cues=["The list's name or purpose before or after list, such as grocery, shopping or to do.",
              "The item being added or removed is not the list name."],
        formats=["The spoken name words, one to five words."],
        ambiguity="An item and a list can both be nouns; the list is the container named with list, the item is "
                  "what goes in or out.",
        limitation="The word list is kept inside some spans and outside others; do not treat either convention as "
                   "a rule."),
    "meal_type": dict(
        roles={
            "calendar_query": "the meal of the scheduled event",
            "calendar_remove": "the meal date to delete",
            "calendar_set": "the meal of the event to schedule",
            "email_sendemail": "the meal mentioned in the email",
            "general_quirky": "a meal mentioned in conversation",
            "lists_remove": "the meal entry to remove",
            "recommendation_events": "the meal a place is sought for",
            "recommendation_locations": "the meal a place is sought for",
            "takeaway_order": "the meal to order for",
            "weather_query": "the meal the weather affects",
        },
        cues=["A meal word, breakfast, lunch or dinner, also inside phrases such as dinner date or lunch with."],
        formats=["One spoken word."],
        ambiguity="Lunch or dinner can name an event; annotate the meal word as meal_type and leave other event "
                  "words to event_name.",
        limitation="Thirty mentions of three meal words; other meal words are unattested here."),
    "media_type": dict(
        roles={
            "calendar_set": "the medium or channel of the event",
            "email_querycontact": "the social network of the contact",
            "general_quirky": "a medium named in conversation",
            "music_settings": "the media unit whose playback repeats",
            "news_query": "the news channel or outlet",
            "play_audiobook": "the medium, such as audiobook or book",
            "play_music": "the service or channel to play",
            "play_radio": "the medium or channel to tune",
            "recommendation_movies": "the medium, such as movie",
            "social_post": "the social medium to post on",
            "social_query": "the social medium to check",
        },
        cues=["A medium, channel or service name, such as facebook, tweet, instagram, audiobook or a news channel.",
              "A title is never the medium; keep a show or book title in its own slot."],
        formats=["The spoken name, usually one word; a verb such as tweet is the medium when it names the channel."],
        ambiguity="A service can also be an app_name or a business_name; choose media_type when it names the "
                  "medium of the content.",
        limitation="Generic words such as music or song are labeled media_type in some support and not in others; "
                   "do not treat either as a rule."),
    "movie_name": dict(
        roles={
            "general_quirky": "the film whose story is asked",
            "play_music": "a title to play, annotated as a film",
            "qa_factoid": "the film a fact is asked about",
            "recommendation_movies": "the film to review or recommend",
        },
        cues=["A film title after the movie, is or how is, or as the subject of a rating question."],
        formats=["The spoken title as uttered, one to six words."],
        ambiguity="A genre phrase is movie_type; a music album title can be annotated as a film, so check what is "
                  "being played.",
        limitation="Music titles are labeled movie_name in some selected support and a genre phrase is labeled a "
                   "title; do not reproduce these labels."),
    "movie_type": dict(
        roles={
            "recommendation_events": "the film genre sought in an event request",
            "recommendation_movies": "the genre or kind of film to recommend",
        },
        cues=["A film genre or kind, such as romantic comedy, sci fi or blockbuster."],
        formats=["The spoken genre words, one or two words, with a quality adjective only when spoken inside it."],
        ambiguity="A performer's name can be annotated here in error; a named film is movie_name.",
        limitation="Eight mentions; one selected span is a performer's name and only one genre span keeps an "
                   "adjective; do not copy these as rules."),
    "music_descriptor": dict(
        roles={
            "music_dislikeness": "the description of the disliked music",
            "music_likeness": "the description of the liked music",
            "music_settings": "the collection whose playback changes",
            "play_music": "the description of the music to play",
            "play_radio": "the mood or activity for the station",
        },
        cues=["A descriptive word for music that is neither a genre nor a title, such as popular, sad, background "
              "or workout."],
        formats=["The spoken words, one to three words."],
        ambiguity="Genre words are music_genre and named artists are artist_name, although the support labels one "
                  "artist name as a descriptor.",
        limitation="Nine mentions; one artist name and one library phrase are labeled descriptors, so the boundary "
                   "with genre and artist is not fixed."),
    "music_genre": dict(
        roles={
            "lists_query": "the genre of the playlist asked about",
            "music_likeness": "the genre the user likes",
            "music_query": "the genre asked about",
            "music_settings": "the genre whose playback repeats",
            "play_music": "the genre to play",
            "play_radio": "the genre of the station",
        },
        cues=["A genre word such as rap, rock, jazz, classical or hip hop, before music, playlist or station."],
        formats=["The spoken genre words, one or two words."],
        ambiguity="A station call sign or a descriptor can be labeled genre in error; choose music_genre for a "
                  "style of music.",
        limitation="Eighteen mentions; one station call sign is labeled a genre, and when two genres are spoken in "
                   "a change only one may be annotated."),
    "news_topic": dict(
        roles={
            "general_quirky": "a topic mentioned in conversation",
            "news_query": "the topic of the news wanted",
            "qa_stock": "the market topic of the update",
            "social_query": "the topic of articles on social media",
        },
        cues=["The subject of news or updates, after news on, about or regarding, or before news."],
        formats=["The spoken topic words, one to three words, keeping a possessive as spoken."],
        ambiguity="A person or place in the topic can be its own slot; keep news_topic to the subject phrase.",
        limitation="Eleven mentions; a named person in the news is labeled person rather than topic, so the split "
                   "is not fixed."),
    "order_type": dict(
        roles={
            "alarm_set": "the delivery event an alarm follows",
            "takeaway_order": "the fulfillment mode of the order",
            "takeaway_query": "the fulfillment mode asked about",
        },
        cues=["A fulfillment word, such as delivery, takeaway, take out, takeout, carryout or pickup."],
        formats=["The spoken word or two as uttered, with spoken spellings kept."],
        ambiguity="Home delivery keeps only the mode word; the food and the restaurant go to their own slots.",
        limitation="Fifteen mentions; one appears in an alarm request, so the type is not exclusive to takeaway "
                   "intents."),
    "person": dict(
        roles={
            "calendar_query": "the person the scheduled event involves",
            "calendar_remove": "the person of the event to delete",
            "calendar_set": "the person the event is with",
            "datetime_convert": "the person whose local time is asked",
            "email_addcontact": "the contact to add or update",
            "email_query": "the sender of the emails",
            "email_querycontact": "the contact looked up",
            "email_sendemail": "the recipient of the email",
            "general_joke": "the person the joke is for",
            "general_quirky": "a person discussed in conversation",
            "lists_createoradd": "the person to add to a list",
            "lists_query": "a person named with a list item",
            "lists_remove": "the person to remove from a list",
            "music_query": "a person whose music is discussed",
            "news_query": "the person the news concerns",
            "play_podcasts": "the host or subject of the podcast",
            "play_radio": "the radio host to play",
            "qa_factoid": "the person a fact is asked about",
            "qa_stock": "the investor a stock question concerns",
            "social_post": "the person addressed or named in a post",
            "social_query": "the person whose posts are checked",
            "transport_query": "the person whose place is the destination",
            "transport_taxi": "the person the taxi picks up",
        },
        cues=["A personal name, first or full, often after with, from, to or about; a possessive stays as spoken.",
              "A role word such as mom is relation, not person, even when it identifies someone."],
        formats=["The spoken name words, one to three words, with a title such as mr kept when spoken."],
        ambiguity="Two named people are separate mentions even when the same name repeats; a name next to a role "
                  "keeps each in its own slot.",
        limitation="Assignments between person and artist_name are inconsistent, and a sports team and a store are "
                   "labeled person; do not copy these assignments."),
    "personal_info": dict(
        roles={
            "calendar_set": "the department of a meeting attendee",
            "email_addcontact": "the contact field to add or change",
            "email_query": "the contact field sought in emails",
            "email_querycontact": "the contact field asked for",
            "email_sendemail": "the address field to send to",
            "recommendation_events": "the user detail locating the events",
            "social_post": "the account detail to post or find",
        },
        cues=["The name of a contact field, such as email, number, address or office number, not its value.",
              "The value of an email address is email_address; the field name is personal_info."],
        formats=["The spoken field words, one to three words."],
        ambiguity="Distinguish the field name from its value; a verb such as live can stand for an address field.",
        limitation="Eighteen mentions; a department and a verb are labeled fields, so the boundary is not fixed."),
    "place_name": dict(
        roles={
            "calendar_query": "the venue or city of scheduled events",
            "calendar_remove": "the place of the event to delete",
            "calendar_set": "the venue of the event or reminder",
            "datetime_convert": "a place whose local time is converted",
            "datetime_query": "the place whose time or date is asked",
            "email_querycontact": "the place a contact lives or comes from",
            "email_sendemail": "a place named in the email",
            "general_quirky": "a place mentioned in conversation",
            "iot_cleaning": "the surface or area to clean",
            "lists_createoradd": "the store a list is for",
            "lists_remove": "the store of the list to delete",
            "news_query": "the place the news covers",
            "play_podcasts": "the place a podcast topic concerns",
            "qa_currency": "the country whose currency is asked about",
            "qa_factoid": "the place a fact is asked about",
            "recommendation_events": "the area to find events in",
            "recommendation_locations": "the area to search",
            "recommendation_movies": "the area or cinema of the showings",
            "social_post": "the place to check in to",
            "transport_query": "the origin or destination of directions",
            "transport_taxi": "the pickup point or destination",
            "transport_ticket": "the departure point or destination",
            "transport_traffic": "the area or the route endpoints",
            "weather_query": "the place of the forecast",
        },
        cues=["A named or generic place, such as a city, country, venue, landmark, home or airport, often after in, "
              "at, to or from.",
              "Routes and conversions can contain two places; keep each endpoint as its own mention in spoken "
              "order."],
        formats=["The spoken place words as uttered, usually one or two words."],
        ambiguity="A time zone abbreviation can be labeled place_name or time_zone; a room inside the home is "
                  "house_place.",
        limitation="Spans vary in scope (an office word kept or dropped, a whole distance phrase), and some zone "
                   "abbreviations are labeled places; do not reproduce these as rules."),
    "player_setting": dict(
        roles={
            "iot_hue_lightchange": "a shuffle mode applied to light colors",
            "music_likeness": "the replay or repeat of a liked song",
            "music_query": "the playback mode asked about",
            "music_settings": "the playback mode to set",
            "play_audiobook": "the playback action, such as resume",
            "play_music": "the playback mode while playing",
            "play_podcasts": "the playback action for the episode",
        },
        cues=["A playback mode or transport word, such as shuffle, repeat, resume, continue, restart, skip, rewind "
              "or forward.",
              "The thing being played stays in its own slot."],
        formats=["The spoken words, mostly one word; a longer phrase is kept when spoken as one setting."],
        ambiguity="A playing verb can be the request itself or a setting; annotate it when it names a mode or a "
                  "resume-like action.",
        limitation="Span scope varies from one word to a whole clause; do not treat the longer spans as a rule."),
    "playlist_name": dict(
        roles={
            "music_likeness": "the playlist to save to",
            "play_music": "the playlist to play",
        },
        cues=["The name of a playlist, before or after playlist or list."],
        formats=["The spoken name words, one to five words."],
        ambiguity="A genre word naming a playlist is the playlist name in that request.",
        limitation="Four mentions over two intents; the word list is inside one span only."),
    "podcast_descriptor": dict(
        roles={"play_podcasts": "which episode to play, such as next or latest"},
        cues=["An episode selector such as next episode or latest podcast episode, or the topic of the podcast."],
        formats=["The spoken words, one to three words."],
        ambiguity="A named show is podcast_name; the descriptor selects among episodes.",
        limitation="Five mentions under one intent; one topic word is labeled a descriptor."),
    "podcast_name": dict(
        roles={"play_podcasts": "the podcast show to play"},
        cues=["The show's name after podcast by, of or on."],
        formats=["The spoken name, one or two words."],
        ambiguity="A topic after on can be labeled the name, and a person's name can be the host instead.",
        limitation="Three mentions under one intent; a topic word is labeled a name."),
    "radio_name": dict(
        roles={"play_radio": "the station or show to tune to"},
        cues=["A station name or call sign, or the name of a radio show, after tune in to, turn the radio to or "
              "play."],
        formats=["The spoken words as uttered, numbers in words, one to five words."],
        ambiguity="A radio host is person in some support and part of a show name in others.",
        limitation="Six mentions under one intent; host names are split between person and radio_name."),
    "relation": dict(
        roles={
            "alarm_remove": "a relation named in the request",
            "alarm_set": "the group or role the meeting is with",
            "calendar_query": "the role, such as doctor, of the appointment",
            "calendar_remove": "the relation of the event to delete",
            "calendar_set": "the role or relation the event is with",
            "email_addcontact": "the relation whose contact changes",
            "email_query": "the relation who sent the emails",
            "email_querycontact": "the relation whose details are asked",
            "email_sendemail": "the relation to email",
            "general_joke": "the role or group a joke is about",
            "general_quirky": "a relation mentioned in conversation",
            "lists_remove": "the relation to remove from a list",
            "play_radio": "the audience the songs are for",
            "social_post": "the relation a post addresses",
            "social_query": "the relation whose posts are checked",
            "transport_query": "the relation whose home is the destination",
            "transport_ticket": "the relation the ticket is for",
        },
        cues=["A family, social or professional role word, such as mom, dad, friend, boss, dentist or doctor.",
              "A name next to the role word is person; keep both spans apart."],
        formats=["The spoken role words, one or two words, with an article or possessive kept as spoken."],
        ambiguity="A role word can identify one person or a group; do not replace it with a name the utterance "
                  "does not speak.",
        limitation="One organization acronym is labeled a relation, and spoken articles are kept in some spans "
                   "only; do not copy these as rules."),
    "song_name": dict(
        roles={
            "lists_remove": "the song to delete",
            "music_likeness": "the song to save",
            "music_query": "the song asked about",
            "music_settings": "the song whose playback changes",
            "play_music": "the song to play",
        },
        cues=["A song title after play, the song or song, and before by and an artist."],
        formats=["The spoken title as uttered, with contractions split as spoken, one to six words."],
        ambiguity="A reference such as last song can be annotated as the title; keep the artist after by in "
                  "artist_name.",
        limitation="Nine mentions; a relative reference is labeled a title; do not treat it as a rule."),
    "sport_type": dict(
        roles={
            "calendar_query": "the team or game asked about",
            "calendar_set": "the game to be reminded of",
        },
        cues=["A sport or game phrase in a schedule request, possibly with a team name inside it."],
        formats=["The spoken words, one to three words, with game kept when spoken."],
        ambiguity="A team name alone is labeled here; a match played on a device is game_name.",
        limitation="Four mentions over two intents and one team; the support is too narrow for general rules."),
    "time": dict(
        roles={
            "alarm_query": "the ring time or window checked",
            "alarm_remove": "the ring time of the alarm to cancel",
            "alarm_set": "the ring time or delay of the alarm",
            "audio_volume_mute": "how long to stay muted",
            "calendar_query": "the time or window of the schedule",
            "calendar_remove": "the time of the event to delete",
            "calendar_set": "the time of the event or reminder",
            "datetime_convert": "the clock time to convert",
            "datetime_query": "the time or period asked about",
            "email_query": "the window in which emails arrived",
            "email_sendemail": "a time in the email or the reply rule",
            "general_quirky": "a time mentioned in conversation",
            "iot_cleaning": "when the cleaning starts",
            "iot_coffee": "when the coffee is made",
            "iot_hue_lightchange": "when the color change happens",
            "iot_hue_lightoff": "when or how soon the lights go off",
            "lists_createoradd": "the period of the list",
            "news_query": "the time of the news or the update",
            "play_audiobook": "when, or after what delay, to play",
            "play_music": "when the song plays",
            "play_podcasts": "the time or position in the episode",
            "play_radio": "when the station plays",
            "qa_factoid": "the period a fact covers",
            "qa_stock": "the period of the market question",
            "recommendation_events": "the time or weekend of the events",
            "recommendation_locations": "the opening time asked about",
            "recommendation_movies": "the showing time",
            "social_query": "the window of social activity",
            "takeaway_order": "when the order is delivered",
            "transport_query": "the time of the train or the window",
            "transport_taxi": "the pickup time",
            "transport_ticket": "the travel time to book",
            "transport_traffic": "the time or the departure delay",
            "weather_query": "the time window of the forecast",
        },
        cues=["A clock reading, a duration or a time window, such as five pm, midnight, thirty minutes or next two "
              "days.",
              "A part of day such as morning is timeofday and a calendar day is date; when both are spoken with a "
              "clock reading, keep each in its own slot."],
        formats=["The spoken words as uttered, numbers in words, with before, after or in kept when spoken."],
        ambiguity="A duration can be a delay or a length; a range has two endpoints kept as separate mentions.",
        limitation="Some selected spans label a non-temporal word or a refusal to name a time as time, and weekends "
                   "appear as time and as date; do not reproduce these labels."),
    "time_zone": dict(
        roles={
            "alarm_set": "the zone to move the alarms to",
            "datetime_convert": "the source or target zone",
            "datetime_query": "the zone whose time is asked",
        },
        cues=["A zone name or abbreviation word, such as eastern, central, mountain or pacific, usually without "
              "the word time."],
        formats=["One spoken word, without time or standard."],
        ambiguity="Some zone abbreviations are labeled place_name; when two zones are spoken, keep each in spoken "
                  "order.",
        limitation="Nine mentions; abbreviations are split between time_zone and place_name."),
    "timeofday": dict(
        roles={
            "alarm_query": "the part of day of the alarm checked",
            "alarm_remove": "the part of day of the alarm to cancel",
            "alarm_set": "the part of day the alarm rings",
            "audio_volume_mute": "the part of day of a mute request",
            "calendar_query": "the part of day of the schedule",
            "calendar_set": "the part of day of the event",
            "datetime_convert": "a part of day to convert",
            "datetime_query": "the part of day asked about",
            "email_query": "the part of day the emails were sent",
            "email_sendemail": "the part of day in the email",
            "general_quirky": "a part of day in conversation",
            "iot_coffee": "when the coffee is brewed",
            "lists_query": "the part of day of the list",
            "lists_remove": "the part of day of the item to remove",
            "music_query": "the part of day the music played",
            "news_query": "the part of day of the news",
            "play_music": "the part of day of the music",
            "play_podcasts": "the part of day of the episode",
            "play_radio": "the part of day the radio plays",
            "recommendation_events": "the part of day of the events",
            "recommendation_movies": "the part of day of the showings",
            "takeaway_order": "the part of day of the delivery",
            "transport_query": "the part of day of travel",
            "transport_taxi": "the part of day of the ride",
            "transport_ticket": "the part of day of travel to book",
            "transport_traffic": "the part of day of the traffic",
            "weather_query": "the part of day of the forecast",
        },
        cues=["A part of day, such as morning, afternoon, evening, night, tonight or noon.",
              "A clock reading beside it is time; keep each in its own slot."],
        formats=["The spoken word as uttered, mostly one word, with a possessive kept as spoken."],
        ambiguity="Noon and night can be labeled time in some support; prefer timeofday for a named part of day.",
        limitation="The word this is kept in some spans and dropped in others; do not reproduce the convention as a "
                   "rule."),
    "transport_agency": dict(
        roles={"transport_taxi": "the ride-share company to book"},
        cues=["A ride company name after an or a, such as uber."],
        formats=["One spoken word."],
        ambiguity="A ride application can be labeled app_name in other support.",
        limitation="Five mentions of one company; the support is too narrow for general rules."),
    "transport_descriptor": dict(
        roles={
            "transport_query": "the travel action, such as departure",
            "transport_ticket": "the ticket kind or class, such as return",
        },
        cues=["A trip or seat descriptor, such as return, round trip or business class, or a departure verb."],
        formats=["The spoken words, one or two words."],
        ambiguity="A descriptor modifies the trip; the mode of travel is transport_type.",
        limitation="Four mentions over two intents; too few to define the type."),
    "transport_name": dict(
        roles={
            "transport_query": "the named train or service asked about",
            "transport_taxi": "the flight the taxi meets",
            "transport_ticket": "the named train or service to book",
        },
        cues=["A named or numbered service, such as an express, a flight number or a train brand."],
        formats=["The spoken name or number words, one to eight words."],
        ambiguity="A spoken number can be a service number rather than a time or a quantity; keep it with its "
                  "service word.",
        limitation="Six mentions; one flight span includes the word flight and a train number span does not."),
    "transport_type": dict(
        roles={
            "play_podcasts": "the mode of travel asked about",
            "transport_query": "the mode of travel asked about",
            "transport_taxi": "the kind of ride to book",
            "transport_ticket": "the mode of travel to book",
        },
        cues=["A mode of travel noun, such as train, taxi, cab or ride."],
        formats=["One spoken word, singular or plural as spoken."],
        ambiguity="A provider is transport_agency and a named service is transport_name; keep only the mode.",
        limitation="Thirty-seven mentions, mostly train; two train-time requests are labeled play_podcasts, so the "
                   "intent pairing is not a rule."),
    "weather_descriptor": dict(
        roles={"weather_query": "the condition, measure or gear asked about"},
        cues=["A weather condition, a measure or a weather-related item, such as rain, snow, temperature, degrees "
              "or umbrella."],
        formats=["The spoken words, one to three words."],
        ambiguity="Gear and actions can be annotated as descriptors; keep the place and the date in their slots.",
        limitation="Twelve mentions under one intent; span scope varies from one word to a phrase."),
}

# --- slot contracts v2: the eight revised cards (2026-09-17 revisions review, section 3) ---------------------------
# Each gains one extraction cue (role, the intents the pool pairs the type with, the span convention). Seven gain an
# emit-when-spoken sentence placed in front of the unchanged missing behaviour; time gains the spoken-form rule and
# one admissible format instead.

SLOT_REVISIONS_V2 = {
    "date": dict(
        cue="Role: the calendar anchor of the request. The pool pairs date with weather_query, datetime_query, "
            "calendar_query, news_query, recommendation_events, calendar_set, calendar_remove, alarm_set, "
            "alarm_remove and transport_ticket among forty intents; relative expressions (today, this weekend) and "
            "absolute ones (fifteenth march) are both annotated as date whenever spoken, mostly as one or two words.",
        emit="When a calendar expression is spoken, emit it as date, relative or absolute, even when the request "
             "does not depend on it."),
    "time": dict(
        cue="Role: the clock reading, duration or window. The pool pairs time most often with alarm_set and "
            "calendar_set among thirty-four intents. Spoken-form rule: no annotated span uses a digit or a colon, "
            "and a spoken am or pm stays a separate word; write the spoken words exactly (five pm, five thirty am, "
            "nine pm, midnight, eight twenty), never a numeric clock.",
        format="A clock time in spoken words with a separate am or pm when spoken, never digits or a colon "
               "(five thirty am)."),
    "media_type": dict(
        cue="Role: the medium, channel or service, never the content title. The pool pairs media_type with "
            "social_post, social_query, play_audiobook, news_query, play_radio, play_music and calendar_set among "
            "eleven intents; spans are one word in almost all support (facebook, tweet).",
        emit="When a medium, channel or service is spoken, emit it as media_type even when the intent already "
             "implies it."),
    "transport_type": dict(
        cue="Role: the mode of travel, not the provider, the seat class or the destination. The pool pairs "
            "transport_type with transport_ticket, transport_query, transport_taxi and play_podcasts; every span is "
            "one word (train, taxi).",
        emit="When a mode of travel is spoken, emit it as transport_type even beside a ticket or a destination."),
    "business_type": dict(
        cue="Role: the category noun of the place, with its spoken modifiers. The pool pairs business_type with "
            "recommendation_locations, takeaway_order, takeaway_query, iot_coffee, recommendation_movies, "
            "recommendation_events and transport_query among eleven intents; spans are mostly one word "
            "(restaurant), otherwise two (coffee shop).",
        emit="When a category of place is spoken, emit it as business_type even beside a business name."),
    "order_type": dict(
        cue="Role: the fulfillment mode itself. The pool pairs order_type with takeaway_query, takeaway_order and "
            "alarm_set; spans are almost always one word (delivery, takeaway), two only for take out.",
        emit="When a fulfillment mode is spoken, emit it as order_type even when the request is about the food."),
    "relation": dict(
        cue="Role: the role word with its possessive ending when spoken (mom 's), separate from any adjacent name. "
            "The pool pairs relation with email_sendemail, calendar_set, email_query, email_querycontact and "
            "general_quirky among seventeen intents; spans are mostly one word, otherwise two (old friend).",
        emit="When a role word is spoken, emit it as relation even when a name follows it."),
    "event_name": dict(
        cue="Role: the occasion or activity. The pool names it most often under calendar_set and calendar_query, "
            "also under calendar_remove, recommendation_events and alarm_set among twenty-one intents; spans run "
            "from one word (meeting) to five, keeping the spoken modifiers.",
        emit="When an occasion or activity is spoken, emit it as event_name, including under calendar_set and "
             "calendar_query."),
}


# --- assembly ------------------------------------------------------------------------------------------------------

def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dump(document: object) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def roster_identity(path: Path) -> str:
    return json.loads(path.read_bytes())["split_identity_hash"]


def intent_rows(support: dict) -> list[dict]:
    groups = collections.defaultdict(list)
    for row in support["intent_support"]:
        groups[row["intent_id"]].append(row)
    if set(groups) != set(INTENTS):
        raise SystemExit("intent inventory differs from the support")
    rows = []
    for intent_id, (activation, neighbour, distinction, special) in sorted(INTENTS.items()):
        members = groups[intent_id] + groups[neighbour]
        rows.append({
            "intent_id": intent_id,
            "activation_boundary": [activation],
            "confusable_exclusions": [{"neighboring_intent_id": neighbour, "distinction": distinction}],
            "uncertainty_behavior": INTENT_UNCERTAINTY,
            "limitations": [INTENT_SHARED_LIMITATION, *([special] if special else [])],
            "source_support": {
                "source_sample_ids": sorted({rid for row in members for rid in row["recording_ids"]}),
                "provenance_refs": sorted({f"semantic-authoring-support:intent:{row['utterance_id']}"
                                           for row in members}),
            },
        })
    return rows


def slot_rows(support: dict, *, revised: bool) -> list[dict]:
    carriers = collections.defaultdict(list)
    pairs = collections.defaultdict(set)
    for row in support["slot_support"]:
        for entity in row["entities"]:
            carriers[entity["entity_type"]].append(row)
            pairs[entity["entity_type"]].add(row["intent_id"])
    if set(carriers) != set(SLOTS):
        raise SystemExit(f"slot inventory differs from the support: {sorted(set(carriers) ^ set(SLOTS))}")
    rows = []
    for slot_id, card in sorted(SLOTS.items()):
        if set(card["roles"]) != pairs[slot_id]:
            raise SystemExit(f"{slot_id}: bindings {sorted(card['roles'])} differ from pairs {sorted(pairs[slot_id])}")
        cues, formats, missing = list(card["cues"]), list(card["formats"]), SLOT_MISSING
        revision = SLOT_REVISIONS_V2.get(slot_id) if revised else None
        if revision:
            cues.append(revision["cue"])
            if "emit" in revision:
                missing = f"{revision['emit']} {SLOT_MISSING}"
            if "format" in revision:
                formats.append(revision["format"])
        members = carriers[slot_id]
        rows.append({
            "slot_id": slot_id,
            "intent_bindings": [{"intent_id": intent_id, "semantic_role": role}
                                for intent_id, role in sorted(card["roles"].items())],
            "extraction_cues": cues,
            "admissible_formats": formats,
            "missing_behavior": missing,
            "ambiguity_behavior": card["ambiguity"],
            "limitations": [SLOT_SHARED_LIMITATION, card["limitation"]],
            "source_support": {
                "source_sample_ids": sorted({rid for row in members for rid in row["recording_ids"]}),
                "provenance_refs": sorted({f"semantic-authoring-support:slot:{row['utterance_id']}"
                                           for row in members}),
            },
        })
    return rows


def check_identifiers(rows: list[dict], valid: set[str]) -> None:
    """Every underscore identifier a card names must be a real intent id or entity type."""

    for row in rows:
        text = json.dumps({key: value for key, value in row.items() if key != "source_support"})
        for word in re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", text):
            if word not in valid and word not in {"intent_id", "semantic_role", "activation_boundary",
                                                  "confusable_exclusions", "neighboring_intent_id",
                                                  "uncertainty_behavior", "slot_id", "intent_bindings",
                                                  "extraction_cues", "admissible_formats", "missing_behavior",
                                                  "ambiguity_behavior"}:
                raise SystemExit(f"card names an unknown identifier {word!r}")


def check_against_receipt(repo: Path) -> None:
    receipt = json.loads((repo / CONTRAST_RECEIPT).read_bytes())["cards"]
    for intent_id, (_, neighbour, _, special) in INTENTS.items():
        card = receipt[intent_id]
        if card["authored_exclusion_named"] != [neighbour]:
            raise SystemExit(f"{intent_id}: neighbour {neighbour} differs from the receipt")
        if card["card_specific_limitations_kept"] != (1 if special else 0):
            raise SystemExit(f"{intent_id}: card-specific limitation count differs from the receipt")


def check_time_spans(support: dict) -> dict:
    spans = [entity["filler"] for row in support["slot_support"] for entity in row["entities"]
             if entity["entity_type"] == "time"]
    facts = {"mentions": len(spans), "with_digit": sum(bool(re.search(r"[0-9]", s)) for s in spans),
             "with_colon": sum(":" in s for s in spans),
             "with_spoken_am_pm": sum(bool(re.search(r"\b(am|pm)\b", s)) for s in spans)}
    if facts != {"mentions": 131, "with_digit": 0, "with_colon": 0, "with_spoken_am_pm": 67}:
        raise SystemExit(f"time evidence differs from the revisions review: {facts}")
    for form in ("five pm", "five thirty am", "nine pm", "midnight", "eight twenty"):
        if form not in spans:
            raise SystemExit(f"time surface form {form!r} is not attested in the pool")
    return facts


def write_create_only(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.read_bytes() != raw:
            raise SystemExit(f"refusing to overwrite {path}")
        return
    path.write_bytes(raw)


def main() -> int:
    data_root, repo, out = map(Path, sys.argv[1:4])
    support = json.loads((data_root / SUPPORT_RELPATH).read_bytes())["output"]
    check_against_receipt(repo)
    time_facts = check_time_spans(support)
    intents = intent_rows(support)
    slots_v1 = slot_rows(support, revised=False)
    slots_v2 = slot_rows(support, revised=True)
    valid = set(INTENTS) | set(SLOTS)
    for rows in (intents, slots_v1, slots_v2):
        check_identifiers(rows, valid)
    documents = {
        "authored-intent-boundaries-v1.json": {
            "schema": SOURCE_SCHEMA, "builder_policy_id": INTENT_POLICY,
            "source_split_identity_hash": roster_identity(data_root / INTENT_ROSTER), "rows": intents},
        "authored-slot-contracts-v1.json": {
            "schema": SOURCE_SCHEMA, "builder_policy_id": SLOT_POLICY,
            "source_split_identity_hash": roster_identity(data_root / SLOT_ROSTER), "rows": slots_v1},
        "authored-slot-contracts-v2.json": {
            "schema": SOURCE_SCHEMA, "builder_policy_id": SLOT_POLICY,
            "source_split_identity_hash": roster_identity(data_root / SLOT_ROSTER), "rows": slots_v2},
    }
    out.mkdir(parents=True, exist_ok=True)
    pins = {}
    for name, document in documents.items():
        raw = dump(document)
        write_create_only(out / name, raw)
        pins[name] = sha256(raw)
    changed = sorted(a["slot_id"] for a, b in zip(slots_v1, slots_v2, strict=True) if a != b)
    if changed != sorted(SLOT_REVISIONS_V2):
        raise SystemExit(f"v2 changed set {changed} differs from the declared revision set")
    changes = {
        "schema": "saea-port-semantic-authoring-changes-v1",
        "authoring": "root-authored from the governed TRAIN support rows; no model, tokenizer, encoder, scorer, "
                     "GPU/NPU or tool contact",
        "support": {"relpath": SUPPORT_RELPATH, "sha256": sha256((data_root / SUPPORT_RELPATH).read_bytes())},
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "sources": pins,
        "slot_v2_changed_cards": changed,
        "time_evidence": time_facts,
    }
    write_create_only(out / "CHANGES.json", dump(changes))
    print(json.dumps(changes, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
