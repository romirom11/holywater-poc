---
version: 2
---

## dialogue

[Continuity] $continuity. Style: $style. Vertical 9:16, single continuous take, no cuts, no text on screen, no subtitles.
[Timeline] ${lead_in}${start_s}–${end_s}s: $speaker says "$line" ($emotion); the speaker's mouth moves only during this line, no other speech.${tail}
[Audio] $language_name speech, clean lip-synced dialogue, the exact words "$line" and nothing else; $ambient_audio; no music.$strict_block

## lead_in

0–${start_s}s: $visual_prompt, $camera.

## lead_in_at_start

$visual_prompt, $camera.

## tail

${end_s}–${duration_s}s: the shot holds, $speaker stays silent.

## no_dialogue

[Continuity] $continuity. Style: $style. Vertical 9:16, single continuous take, no cuts, no text on screen, no subtitles.
[Timeline] 0–${duration_s}s: $visual_prompt, $camera. No one speaks.
[Audio] no dialogue, no speech, no voice; $ambient_audio; no music.

## strict

 The spoken words must be exactly "$line" — do not add, drop, translate or rephrase any word.
