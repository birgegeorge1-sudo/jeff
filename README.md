# Discord + Mistral + MySQL

This handoff adds Discord as the operator UI and upgrades the conversation engine.

## Discord workflow

Invite your bot to a private server/channel and keep that channel private.

Commands:

```text
!newconvo Esther
!contconvo Esther
!active
!reply Hey, how's your day going?
```

**Important semantics**

- `!newconvo Esther` means create/activate Esther's conversation.
- `!contconvo Esther` means continue an existing Esther conversation.
- `!contconvo` never silently creates a record.
`!deleteconvo` permanently removes the contact and cascaded data; deleting the active contact clears the active pointer.
- The active contact is the only memory context supplied to Mistral.

For a real incoming-message adapter, connect an approved messaging platform to the HTTP core or add an adapter that converts an inbound event to the same `contact + message` structure.

## Discord setup

In the Discord Developer Portal enable the Message Content Intent for the bot, then put the bot token into:

```env
DISCORD_BOT_TOKEN=...
OWNER_DISCORD_USER_ID=...
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python run_discord.py
```

## Conversation engine

The engine now evaluates:

- boredom / flatness
- reciprocity
- vibe
- topic repetition
- conversational mode
- positive signals
- concerns
- date readiness
- intimacy readiness
- voice-note suitability
- memory updates

The model is instructed to avoid turning the conversation into an interview. Drafts are written in Jefferey's voice and do not contain bot/AI framing, but the system does not autonomously impersonate Jefferey or fabricate answers to direct identity questions. A useful rhythm is:

```text
React → share → curiosity → play → deepen → lighten → space
```

It chooses among these dynamically.

## Romance and intimacy

The system treats sexuality as a normal possible stage of an adult relationship, not as a target to force.

When conversation becomes romantic/sexual, the analysis looks for:

- mutual participation
- comfort
- reciprocal escalation
- explicit boundaries
- willingness to continue

It can suggest wording that checks comfort rather than guessing. It should never pressure someone, seek sexual images, or treat silence as consent.

## Persona

`persona.json` contains the user's supplied facts and a style guide.

Verified facts are separated from style inference so the model can sound consistent without hallucinating a biography.

For the Tennessee influence, the copilot uses relaxed Southern warmth and cadence without fake dialect spelling.

## Database isolation

`contacts`, `messages`, and `memories` are keyed by `contact_id`. The engine receives only:

1. active contact
2. active contact's memories
3. active contact's recent messages
4. current inbound message

This is the primary defense against cross-contact context leakage.

## Current limitation

Discord is the operator console. Discord is not pretending to be a dating partner. The social platform where the women are actually messaging you still requires a separate platform adapter using that service's supported API/authentication.

The design intentionally keeps message generation separate from sending so you can inspect the draft before it leaves your account.
